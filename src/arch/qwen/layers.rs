// layers.rs
//
// SEAM — Path 2 (in progress, Escopo 3). ADR-0002.
//
// This file holds the hand-written Qwen3 building blocks that wrap
// attention.rs and mlp.rs into a full transformer stack: token embedding,
// RMSNorm, one transformer block (input norm -> attention -> residual ->
// post-attention norm -> MLP -> residual), the final norm, and the output
// projection to vocabulary logits.
//
// Peça 1/7 — Embeddings. Dequantizes `token_embd.weight` from the GGUF file
// straight to F32 (matches the oracle: candle_transformers dequantizes on
// load too, since `Embedding` only operates on plain tensors — quantized
// gather isn't a thing in candle). Lookup is a plain row-gather via
// `Tensor::index_select`, mirroring `candle_nn::Embedding::forward`.
//
// Peça 7/7 — the 28-layer loop (`Layer`, `Model`). This is where Path 2
// stops being isolated pieces and becomes the actual forward pass: `Model`
// replaces `candle_transformers::models::quantized_qwen3::ModelWeights` in
// `arch/qwen/mod.rs`. `candle_transformers` remains a dependency (per the
// Escopo 3 briefing, "não remover ainda") but is only referenced from
// `#[cfg(test)]` code from here on — the oracle for validation.

use candle_core::quantized::{QMatMul, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor};

/// Token embedding table: `token_embd.weight` dequantized to F32.
pub struct TokenEmbedding {
    weight: Tensor, // (vocab_size, hidden_size), F32
    hidden_size: usize,
}

impl TokenEmbedding {
    /// Build from the raw quantized tensor read out of the GGUF file.
    pub fn from_qtensor(qtensor: QTensor, hidden_size: usize, device: &Device) -> Result<Self> {
        let weight = qtensor.dequantize(device)?;
        Ok(Self {
            weight,
            hidden_size,
        })
    }

    /// Look up embeddings for `tokens`, shaped `(1, seq_len)` in, returning
    /// `(1, seq_len, hidden_size)`.
    pub fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let (b, l) = tokens.dims2()?;
        let flat = tokens.flatten_all()?;
        let rows = self.weight.index_select(&flat, 0)?;
        rows.reshape((b, l, self.hidden_size))
    }
}

/// Peça 2/7 — RMSNorm. Qwen3 uses RMSNorm (no mean-centering, no bias) at
/// every `attn_norm`/`ffn_norm`/`attn_q_norm`/`attn_k_norm`/`output_norm`
/// site: `x / sqrt(mean(x^2, last_dim) + eps) * weight`.
///
/// Implemented directly on `candle-core` tensor ops (not the fused
/// `candle_nn::ops::rms_norm` kernel the oracle calls) so this piece
/// exercises our own arithmetic rather than delegating to the same
/// primitive being validated against.
pub struct RmsNorm {
    weight: Tensor, // (hidden_size,), F32
    eps: f64,
}

impl RmsNorm {
    /// Build from the raw quantized weight tensor read out of the GGUF file.
    pub fn from_qtensor(qtensor: QTensor, eps: f64, device: &Device) -> Result<Self> {
        let weight = qtensor.dequantize(device)?;
        Ok(Self { weight, eps })
    }

    /// Normalize the last dimension of `x` and scale by `weight`. `x` may
    /// be any shape ending in `hidden_size`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let in_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let last_dim = x.rank() - 1;
        let mean_sq = x.sqr()?.mean_keepdim(last_dim)?;
        let denom = (mean_sq + self.eps)?.sqrt()?;
        let normed = x.broadcast_div(&denom)?;
        normed.broadcast_mul(&self.weight)?.to_dtype(in_dtype)
    }
}

/// One transformer block: input norm -> attention (with its own KV cache)
/// -> residual -> post-attention norm -> MLP -> residual. Owns the KV
/// cache for its own layer (each of the 28 layers caches independently).
pub struct Layer {
    attn_norm: RmsNorm,
    attention: super::attention::Attention,
    ffn_norm: RmsNorm,
    mlp: super::mlp::Mlp,
    cache: super::attention::KvCache,
}

impl Layer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        attn_norm: RmsNorm,
        attention: super::attention::Attention,
        ffn_norm: RmsNorm,
        mlp: super::mlp::Mlp,
    ) -> Self {
        Self {
            attn_norm,
            attention,
            ffn_norm,
            mlp,
            cache: super::attention::KvCache::new(2),
        }
    }

    fn forward(&mut self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let h = self.attn_norm.forward(x)?;
        let h = self.attention.forward(&h, offset, &mut self.cache)?;
        let x = (x + h)?;
        let h2 = self.ffn_norm.forward(&x)?;
        let h2 = self.mlp.forward(&h2)?;
        &x + h2
    }

    fn clear_kv_cache(&mut self) {
        self.cache.reset();
    }
}

/// The full Qwen3 forward pass, hand-written on `candle-core`/`candle-nn`
/// primitives — this is Path 2 (ADR-0002). Reads the same GGUF tensor
/// layout the oracle (`candle_transformers::models::quantized_qwen3::ModelWeights`)
/// does, so it can be built from the exact same file.
///
/// Each layer builds its own `RotaryEmbedding` (cos/sin tables depend only
/// on `head_dim`/`rope_theta`/`max_position_embeddings`, identical across
/// layers) rather than sharing one via `Arc` — a small, one-time,
/// load-time cost, traded for not having to thread a shared table through
/// every layer's constructor.
pub struct Model {
    embed_tokens: TokenEmbedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_head: QMatMul,
}

impl Model {
    pub fn from_gguf<R: std::io::Read + std::io::Seek>(
        content: candle_core::quantized::gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        use super::attention::{Attention, RotaryEmbedding};
        use super::mlp::Mlp;

        let md = |s: &str| match content.metadata.get(s) {
            None => candle_core::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };
        let num_heads = md("qwen3.attention.head_count")?.to_u32()? as usize;
        let num_kv_heads = md("qwen3.attention.head_count_kv")?.to_u32()? as usize;
        let head_dim = md("qwen3.attention.key_length")?.to_u32()? as usize;
        let num_layers = md("qwen3.block_count")?.to_u32()? as usize;
        let hidden_size = md("qwen3.embedding_length")?.to_u32()? as usize;
        let max_position_embeddings = md("qwen3.context_length")?.to_u32()? as usize;
        let rms_norm_eps = md("qwen3.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_theta = md("qwen3.rope.freq_base")?.to_f32()? as f64;

        let embed_tensor = content.tensor(reader, "token_embd.weight", device)?;
        let embed_tokens = TokenEmbedding::from_qtensor(embed_tensor, hidden_size, device)?;

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let prefix = format!("blk.{i}");

            let attn_norm = RmsNorm::from_qtensor(
                content.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?,
                rms_norm_eps,
                device,
            )?;
            let ffn_norm = RmsNorm::from_qtensor(
                content.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?,
                rms_norm_eps,
                device,
            )?;

            let q_proj = QMatMul::from_qtensor(
                content.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?,
            )?;
            let k_proj = QMatMul::from_qtensor(
                content.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?,
            )?;
            let v_proj = QMatMul::from_qtensor(
                content.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?,
            )?;
            let o_proj = QMatMul::from_qtensor(
                content.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?,
            )?;
            let q_norm = RmsNorm::from_qtensor(
                content.tensor(reader, &format!("{prefix}.attn_q_norm.weight"), device)?,
                rms_norm_eps,
                device,
            )?;
            let k_norm = RmsNorm::from_qtensor(
                content.tensor(reader, &format!("{prefix}.attn_k_norm.weight"), device)?,
                rms_norm_eps,
                device,
            )?;
            let attention = Attention::new(
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm,
                k_norm,
                RotaryEmbedding::new(head_dim, max_position_embeddings, rope_theta, device)?,
                num_heads,
                num_kv_heads,
                head_dim,
            );

            let gate_proj = QMatMul::from_qtensor(
                content.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?,
            )?;
            let up_proj = QMatMul::from_qtensor(
                content.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?,
            )?;
            let down_proj = QMatMul::from_qtensor(
                content.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?,
            )?;
            let mlp = Mlp::new(gate_proj, up_proj, down_proj);

            layers.push(Layer::new(attn_norm, attention, ffn_norm, mlp));
        }

        let norm = RmsNorm::from_qtensor(
            content.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
            device,
        )?;
        let lm_head_tensor = match content.tensor(reader, "output.weight", device) {
            Ok(t) => t,
            Err(_) => content.tensor(reader, "token_embd.weight", device)?,
        };
        let lm_head = QMatMul::from_qtensor(lm_head_tensor)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    /// `input`: `(1, new_seq_len)` token ids for this step. `offset`:
    /// position of the first of those tokens in the full sequence. Returns
    /// last-position logits, shape `(1, vocab_size)`.
    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (_b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;
        for layer in &mut self.layers {
            h = layer.forward(&h, offset)?;
        }
        let h = self.norm.forward(&h)?;
        let last_hidden = h.narrow(1, l - 1, 1)?;
        self.lm_head.forward(&last_hidden)?.squeeze(1)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::gguf_file;
    use candle_transformers::models::quantized_qwen3::ModelWeights;
    use std::fs::File;

    const MODEL_PATH: &str = "models/qwen3-0.6b-q8_0.gguf";

    /// Piece 1 validation: our `TokenEmbedding` against the oracle
    /// (`candle_transformers::models::quantized_qwen3::ModelWeights`'s
    /// internal `embed_tokens`, exercised indirectly by feeding the same
    /// token ids through both and comparing the embedding rows).
    ///
    /// We can't reach the oracle's private `embed_tokens` field directly, so
    /// instead we dequantize `token_embd.weight` ourselves via the same
    /// `gguf_file::Content::tensor` API the oracle uses, independently of
    /// our `TokenEmbedding`, and compare row-for-row. This isolates the
    /// embedding lookup from the rest of the oracle's forward pass.
    #[test]
    fn embedding_matches_oracle_dequantization() {
        let device = Device::Cpu;

        let mut file1 = File::open(MODEL_PATH).expect("model file present for validation");
        let content1 = gguf_file::Content::read(&mut file1).expect("valid gguf");
        let hidden_size = content1
            .metadata
            .get("qwen3.embedding_length")
            .expect("hidden size in metadata")
            .to_u32()
            .unwrap() as usize;

        let qtensor = content1
            .tensor(&mut file1, "token_embd.weight", &device)
            .expect("token_embd.weight present");
        let ours = TokenEmbedding::from_qtensor(qtensor, hidden_size, &device).unwrap();

        // Independent oracle-side dequantization: re-open and re-read the
        // same tensor fresh, exactly as `ModelWeights::from_gguf` does.
        let mut file2 = File::open(MODEL_PATH).unwrap();
        let content2 = gguf_file::Content::read(&mut file2).unwrap();
        let oracle_qtensor = content2
            .tensor(&mut file2, "token_embd.weight", &device)
            .unwrap();
        let oracle_weight = oracle_qtensor.dequantize(&device).unwrap();

        let tokens = Tensor::new(&[1u32, 100, 5000, 12], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();

        let ours_out = ours.forward(&tokens).unwrap();
        let oracle_rows = oracle_weight
            .index_select(&tokens.flatten_all().unwrap(), 0)
            .unwrap()
            .reshape((1, 4, hidden_size))
            .unwrap();

        let ours_v = ours_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let oracle_v = oracle_rows.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        assert_eq!(ours_v.len(), oracle_v.len());
        let max_diff = ours_v
            .iter()
            .zip(oracle_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "max diff {max_diff} exceeds threshold 1e-4"
        );

        // Also confirm the oracle model itself loads with the same file, as
        // an end-to-end sanity check that this is really the same tensor
        // layout the running CLI uses (Path 1).
        let mut file3 = File::open(MODEL_PATH).unwrap();
        let content3 = gguf_file::Content::read(&mut file3).unwrap();
        let _ = ModelWeights::from_gguf(content3, &mut file3, &device)
            .expect("oracle loads the same gguf file");
    }

    /// Piece 2 validation: our `RmsNorm` against the oracle's
    /// (`candle_transformers::quantized_nn::RmsNorm`, the fused
    /// `candle_nn::ops::rms_norm` kernel). Feeds a batch of real token
    /// embeddings through `blk.0.attn_norm.weight` on both sides.
    #[test]
    fn rms_norm_matches_oracle() {
        let device = Device::Cpu;

        let mut file = File::open(MODEL_PATH).expect("model file present for validation");
        let content = gguf_file::Content::read(&mut file).expect("valid gguf");
        let hidden_size = content
            .metadata
            .get("qwen3.embedding_length")
            .unwrap()
            .to_u32()
            .unwrap() as usize;
        let eps = content
            .metadata
            .get("qwen3.attention.layer_norm_rms_epsilon")
            .unwrap()
            .to_f32()
            .unwrap() as f64;

        let embed_q = content.tensor(&mut file, "token_embd.weight", &device).unwrap();
        let embed_table = TokenEmbedding::from_qtensor(embed_q, hidden_size, &device).unwrap();

        let norm_q = content
            .tensor(&mut file, "blk.0.attn_norm.weight", &device)
            .unwrap();
        // clone the qtensor's bytes by re-reading, since `from_qtensor` consumes it
        // and we need one copy for ours and one for the oracle.
        let norm_q_oracle = content
            .tensor(&mut file, "blk.0.attn_norm.weight", &device)
            .unwrap();

        let tokens = Tensor::new(&[1u32, 100, 5000, 12], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let x = embed_table.forward(&tokens).unwrap();

        let ours = RmsNorm::from_qtensor(norm_q, eps, &device).unwrap();
        let ours_out = ours.forward(&x).unwrap();

        let oracle = candle_transformers::quantized_nn::RmsNorm::from_qtensor(norm_q_oracle, eps)
            .unwrap();
        let oracle_out = candle_nn::Module::forward(&oracle, &x).unwrap();

        let ours_v = ours_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let oracle_v = oracle_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        assert_eq!(ours_v.len(), oracle_v.len());
        let max_diff = ours_v
            .iter()
            .zip(oracle_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "max diff {max_diff} exceeds threshold 1e-4"
        );
    }

    /// Piece 7 validation: our full `Model` (28-layer Path 2 forward pass)
    /// against the real oracle end-to-end —
    /// `candle_transformers::models::quantized_qwen3::ModelWeights::forward`
    /// — across a prefill step (5 real tokens) followed by a decode step
    /// (1 more token, continuing each side's own KV cache). This is the
    /// integration test the first six pieces were building toward: if
    /// embeddings, RMSNorm, RoPE, GQA attention, SwiGLU MLP and the KV
    /// cache are all individually correct *and* wired together correctly
    /// across 28 layers, the final logits must be close to the oracle's.
    ///
    /// Threshold is **0.24** here, not the 1e-4 pieces 1-6 use (ADR-0003).
    /// `qwen3-0.6b-q8_0.gguf` has no `general.dtype` metadata, so the
    /// oracle's own loader falls back to computing its RoPE cos/sin tables
    /// in F16 (`ModelWeights::from_gguf`'s `None => DType::F16` branch);
    /// ours (piece 3) always computes in F32. That precision gap
    /// accumulates over 28 layers into a ~0.22 max diff on final logits —
    /// a real difference, but in the oracle's own fallback precision, not
    /// a Path 2 logic error (see ADR-0003 for the options considered and
    /// why replicating the oracle's F16 loss was rejected). The magnitude
    /// check alone would be too loose to catch a real wiring bug at that
    /// threshold, so it's paired with an exact argmax check below: the
    /// most-likely next token must match regardless of the relaxed
    /// magnitude tolerance.
    #[test]
    fn model_matches_oracle_end_to_end() {
        let device = Device::Cpu;

        let mut ours_file = File::open(MODEL_PATH).expect("model file present for validation");
        let ours_content = gguf_file::Content::read(&mut ours_file).expect("valid gguf");
        let mut ours = Model::from_gguf(ours_content, &mut ours_file, &device)
            .expect("our Model loads the real gguf file");

        let mut oracle_file = File::open(MODEL_PATH).unwrap();
        let oracle_content = gguf_file::Content::read(&mut oracle_file).unwrap();
        let mut oracle = ModelWeights::from_gguf(oracle_content, &mut oracle_file, &device)
            .expect("oracle loads the same gguf file");

        // Prefill: 5 real token ids.
        let prefill = Tensor::new(&[1u32, 100, 5000, 12, 77], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let ours_logits1 = ours.forward(&prefill, 0).unwrap();
        let oracle_logits1 = oracle.forward(&prefill, 0).unwrap();
        assert_logits_close(&ours_logits1, &oracle_logits1);
        assert_same_argmax(&ours_logits1, &oracle_logits1);

        // Decode: one more token, continuing each side's own KV cache from
        // offset 5.
        let decode = Tensor::new(&[8u32], &device).unwrap().unsqueeze(0).unwrap();
        let ours_logits2 = ours.forward(&decode, 5).unwrap();
        let oracle_logits2 = oracle.forward(&decode, 5).unwrap();
        assert_logits_close(&ours_logits2, &oracle_logits2);
        assert_same_argmax(&ours_logits2, &oracle_logits2);
    }

    /// Piece 7's relaxed magnitude check — see ADR-0003. Threshold 0.24,
    /// not the 1e-4 pieces 1-6 use.
    fn assert_logits_close(ours: &Tensor, oracle: &Tensor) {
        const THRESHOLD: f32 = 0.24;
        assert_eq!(ours.dims(), oracle.dims());
        let a = ours
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let b = oracle
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(a.len(), b.len());
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < THRESHOLD,
            "max diff {max_diff} exceeds threshold {THRESHOLD}"
        );
    }

    /// Independent of the relaxed magnitude threshold above (ADR-0003):
    /// the most-likely next token must be identical between our `Model`
    /// and the oracle. This is what actually guards against a real
    /// 28-layer wiring bug — a logic error could shift several logits by
    /// a large, wrong amount without necessarily tripping the 0.24
    /// magnitude check on this particular prompt, but it would be very
    /// unlikely to leave the argmax untouched too.
    fn assert_same_argmax(ours: &Tensor, oracle: &Tensor) {
        let a = ours
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let b = oracle
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let argmax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                    if x > bv {
                        (i, x)
                    } else {
                        (bi, bv)
                    }
                })
                .0
        };
        assert_eq!(
            argmax(&a),
            argmax(&b),
            "top token diverged between ours and the oracle"
        );
    }
}
