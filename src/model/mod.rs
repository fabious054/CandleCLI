//! Model layer: loading and top-level orchestration.
//!
//! [`CandleModel`] is the concrete handle returned to crate users. It owns
//! the load path — reading a GGUF file (`gguf`), building the tokenizer from
//! the embedded vocabulary, and initializing the Qwen3 architecture
//! (`crate::arch::qwen`) — and drives streaming inference.
//!
//! Per ADR-0002 the forward pass is provided by `candle_transformers`'
//! quantized Qwen3 model, wrapped behind `crate::arch::Architecture`. The
//! sampler is our own (`crate::sampler`).

pub mod gguf;

use std::path::Path;
use std::sync::{Arc, Mutex};

use candle_core::quantized::tokenizer::TokenizerFromGguf;
use candle_core::Device;
use tokenizers::Tokenizer;

use crate::arch::{qwen::Qwen, Architecture};
use crate::sampler::{self, Sampler, SamplerConfig};
use crate::{Error, Result};

/// Upper bound on generated tokens per `infer` call (Escopo 1: fixed).
const MAX_NEW_TOKENS: usize = 512;

/// Contract satisfied by any loaded model, regardless of architecture.
///
/// Kept minimal on purpose; a single architecture (Qwen3) is in scope now,
/// but the trait boundary lets Llama/Phi/Gemma slot in later as new modules.
pub trait Model {
    /// Run streaming inference for `prompt` under `config`.
    fn infer(
        &self,
        prompt: &str,
        config: &SamplerConfig,
    ) -> Result<Box<dyn Iterator<Item = Result<String>>>>;
}

/// Public handle to a model loaded from a GGUF file.
///
/// ```ignore
/// let model = CandleModel::load("qwen3-0.6b.gguf")?;
/// for token in model.infer("prompt", &SamplerConfig::default())? {
///     print!("{}", token?);
/// }
/// ```
///
/// The prompt string is tokenized as-is: any chat template (ChatML for
/// Qwen3) is the caller's responsibility. The CLI applies one via
/// [`crate::session::Session`]; the bare crate API stays unopinionated.
pub struct CandleModel {
    /// Qwen3 forward pass + its KV cache. `Arc<Mutex<_>>` so a streaming
    /// iterator can own a handle while `CandleModel` stays `Send + Sync`.
    arch: Arc<Mutex<Qwen>>,
    /// BPE tokenizer built from the GGUF's embedded vocabulary.
    tokenizer: Arc<Tokenizer>,
    /// Token ids that end generation (EOS, `<|im_end|>`, `<|endoftext|>`).
    stop_ids: Vec<u32>,
}

impl CandleModel {
    /// Load a GGUF model file from disk (CPU only in Escopo 1).
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let gguf::GgufFile {
            content,
            mut reader,
        } = gguf::GgufFile::open(path)?;

        // Tokenizer from the embedded vocabulary — no `tokenizer.json`, no
        // network (ADR-0002 / closed tokenizer decision).
        let tokenizer = Tokenizer::from_gguf(&content)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;

        let mut stop_ids = Vec::new();
        if let Some(id) = content
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32().ok())
        {
            stop_ids.push(id);
        }
        for name in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"] {
            if let Some(id) = tokenizer.token_to_id(name) {
                if !stop_ids.contains(&id) {
                    stop_ids.push(id);
                }
            }
        }

        // Consumes `content` and reads tensors through `reader`.
        let arch = Qwen::from_gguf(content, &mut reader, &Device::Cpu)?;

        Ok(Self {
            arch: Arc::new(Mutex::new(arch)),
            tokenizer: Arc::new(tokenizer),
            stop_ids,
        })
    }

    /// Run streaming inference. Returns an iterator that yields one decoded
    /// text fragment per generated token.
    pub fn infer(
        &self,
        prompt: &str,
        config: &SamplerConfig,
    ) -> Result<Box<dyn Iterator<Item = Result<String>>>> {
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let tokens = encoding.get_ids().to_vec();
        if tokens.is_empty() {
            return Err(Error::Model("prompt vazio após tokenização".into()));
        }

        // Each `infer` call is an independent sequence.
        self.reset_cache();

        Ok(Box::new(TokenStream {
            arch: Arc::clone(&self.arch),
            tokenizer: Arc::clone(&self.tokenizer),
            sampler: sampler::build(config),
            stop_ids: self.stop_ids.clone(),
            tokens,
            offset: 0,
            generated: Vec::new(),
            decoded: String::new(),
            remaining: MAX_NEW_TOKENS,
            done: false,
        }))
    }

    /// Drop the architecture's KV cache (used by `/reset`).
    pub fn reset_cache(&self) {
        if let Ok(mut arch) = self.arch.lock() {
            arch.clear_kv_cache();
        }
    }
}

impl Model for CandleModel {
    fn infer(
        &self,
        prompt: &str,
        config: &SamplerConfig,
    ) -> Result<Box<dyn Iterator<Item = Result<String>>>> {
        CandleModel::infer(self, prompt, config)
    }
}

/// Iterator that drives token-by-token generation.
///
/// On the first `next` it feeds the whole prompt; afterwards it feeds the
/// single previous token, letting the architecture's KV cache carry the
/// context. Decoding is done by re-decoding the full generated sequence and
/// emitting the newly appended suffix — the standard trick for byte-level
/// BPE, where one character may span several tokens.
struct TokenStream {
    arch: Arc<Mutex<Qwen>>,
    tokenizer: Arc<Tokenizer>,
    sampler: Box<dyn Sampler>,
    stop_ids: Vec<u32>,
    /// Full context: prompt tokens followed by generated tokens.
    tokens: Vec<u32>,
    /// KV-cache position = number of tokens already fed.
    offset: usize,
    /// Generated token ids only (for incremental decoding).
    generated: Vec<u32>,
    /// Text decoded from `generated` so far.
    decoded: String,
    /// Generation budget left.
    remaining: usize,
    done: bool,
}

impl Iterator for TokenStream {
    type Item = Result<String>;

    fn next(&mut self) -> Option<Result<String>> {
        if self.done || self.remaining == 0 {
            return None;
        }

        let input: Vec<u32> = if self.offset == 0 {
            self.tokens.clone()
        } else {
            vec![*self.tokens.last().unwrap()]
        };

        let logits = {
            let mut arch = match self.arch.lock() {
                Ok(a) => a,
                Err(_) => {
                    self.done = true;
                    return Some(Err(Error::Model("mutex do modelo envenenado".into())));
                }
            };
            match arch.forward(&input, self.offset) {
                Ok(l) => l,
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            }
        };
        self.offset += input.len();

        let next_id = self.sampler.sample(&logits);
        if self.stop_ids.contains(&next_id) {
            self.done = true;
            return None;
        }

        self.tokens.push(next_id);
        self.generated.push(next_id);
        self.remaining -= 1;

        // `skip_special_tokens = false`: the CLI's `ThinkFilter` needs the
        // literal `<think>` / `</think>` tags (they are special tokens).
        // Stop tokens are handled above, so they never reach here.
        let text = match self.tokenizer.decode(&self.generated, false) {
            Ok(t) => t,
            Err(e) => {
                self.done = true;
                return Some(Err(Error::Tokenizer(e.to_string())));
            }
        };

        // Emit only the newly appended suffix. `get` returns None if the
        // previous length is not on a char boundary yet — emit nothing this
        // step and catch up on the next.
        let fragment = text.get(self.decoded.len()..).unwrap_or("").to_string();
        self.decoded = text;
        Some(Ok(fragment))
    }
}
