//! Qwen3 architecture — Path 1 adapter (ADR-0002).
//!
//! The forward pass is delegated to
//! `candle_transformers::models::quantized_qwen3::ModelWeights`. This module
//! wraps it behind [`crate::arch::Architecture`] and converts between our
//! plain `&[u32]` / `Vec<f32>` interface and candle tensors.
//!
//! Path 2 (a future escopo) replaces this adapter with a hand-written
//! implementation across [`attention`], [`mlp`] and [`layers`]; at that
//! point `quantized_qwen3` becomes a logit oracle for validation. The
//! sub-modules are kept as documented seams — see each file.

pub mod attention;
pub mod layers;
pub mod mlp;

use std::io::{Read, Seek};

use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::quantized_qwen3::ModelWeights;

use crate::arch::Architecture;
use crate::Result;

/// A loaded Qwen3 model (quantized weights + KV cache).
pub struct Qwen {
    weights: ModelWeights,
    device: Device,
}

impl Qwen {
    /// Build from parsed GGUF contents. Consumes `content`; reads tensor
    /// data through `reader`.
    pub fn from_gguf<R: Read + Seek>(
        content: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let weights = ModelWeights::from_gguf(content, reader, device)?;
        Ok(Self {
            weights,
            device: device.clone(),
        })
    }
}

impl Architecture for Qwen {
    fn forward(&mut self, tokens: &[u32], offset: usize) -> Result<Vec<f32>> {
        // (seq_len,) -> (1, seq_len)
        let input = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;
        // ModelWeights::forward returns last-position logits, shape (1, vocab).
        let logits = self.weights.forward(&input, offset)?;
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
        Ok(logits.to_vec1::<f32>()?)
    }

    fn clear_kv_cache(&mut self) {
        self.weights.clear_kv_cache();
    }
}
