// attention.rs
//
// SEAM — Path 2 (in progress, Escopo 3). ADR-0002.
//
// This file holds the hand-written Qwen3 multi-head attention:
// Q/K/V projections, rotary position embedding (RoPE) applied to Q and K,
// grouped-query attention (KV heads repeated to match Q heads), the KV
// cache read/append, scaled dot-product attention, and the output
// projection — built on candle-nn primitives (QMatMul, rotary_emb, softmax).
//
// Peça 3/7 — RoPE. Implemented below (`RotaryEmbedding`), validated in
// isolation against the oracle's fused `candle_nn::rotary_emb::rope` kernel.
//
// Peça 6/7 — KV Cache. Implemented below (`KvCache`), validated against
// the oracle's `candle_nn::kv_cache::ConcatKvCache`.
//
// The rest of the forward pass still runs through the candle_transformers
// adapter in arch/qwen/mod.rs (Path 1). When Path 2 lands, this file
// replaces part of that adapter and the candle_transformers Qwen3 model
// becomes a logit oracle for validation.

use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Module, Result, Tensor, D};

use super::layers::RmsNorm;

/// Rotary position embedding — precomputes `cos`/`sin` tables up to
/// `max_position_embeddings` and applies the "half-rotate" convention:
/// each head-dim vector is split into two halves, the second half negated
/// and swapped with the first, then blended with cos/sin — matching
/// candle_nn::rotary_emb's non-interleaved (`rope`/`rope_slow`) convention,
/// which is what the Qwen3 GGUF weights (and the oracle) expect.
///
/// Implemented directly on tensor ops (not `candle_nn::rotary_emb::rope`,
/// the fused kernel the oracle calls) to exercise our own arithmetic.
pub struct RotaryEmbedding {
    cos: Tensor, // (max_seq_len, head_dim / 2)
    sin: Tensor, // (max_seq_len, head_dim / 2)
}

impl RotaryEmbedding {
    pub fn new(
        head_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        device: &Device,
    ) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let half_d = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half_d), device)?;
        let t = Tensor::arange(0u32, max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?; // (max_seq_len, half_d)
        Ok(Self {
            cos: freqs.cos()?,
            sin: freqs.sin()?,
        })
    }

    /// Rotate the second half of the last dim into the first, negated:
    /// `[x1, x2] -> [-x2, x1]`. Mirrors `candle_nn::rotary_emb::rotate_half`.
    fn rotate_half(x: &Tensor) -> Result<Tensor> {
        let last_dim = x.dim(D::Minus1)?;
        let x1 = x.narrow(D::Minus1, 0, last_dim / 2)?;
        let x2 = x.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
        Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
    }

    /// Apply RoPE to `q` and `k`, both shaped `(batch, heads, seq_len,
    /// head_dim)`, starting at sequence position `offset`.
    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        // (seq_len, half_d) -> (seq_len, head_dim) by duplicating, then
        // broadcastable as (1, 1, seq_len, head_dim).
        let cos = Tensor::cat(&[&cos, &cos], D::Minus1)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let sin = Tensor::cat(&[&sin, &sin], D::Minus1)?
            .unsqueeze(0)?
            .unsqueeze(0)?;

        let q_embed = (q.broadcast_mul(&cos)? + Self::rotate_half(q)?.broadcast_mul(&sin)?)?;
        let k_embed = (k.broadcast_mul(&cos)? + Self::rotate_half(k)?.broadcast_mul(&sin)?)?;
        Ok((q_embed, k_embed))
    }
}

/// Peça 4/7 — Attention. Multi-head self-attention with Grouped Query
/// Attention (GQA): Qwen3-0.6B has 16 query heads sharing 8 key/value
/// heads, each KV head repeated `num_heads / num_kv_heads` times to match
/// query heads before the dot product.
///
/// No KV cache yet (piece 6) — `forward` runs a full self-attention over
/// whatever `x` it is given; `offset` only shifts the RoPE and causal-mask
/// position, matching how a cache-backed call would behave for the newly
/// appended slice.
///
/// Own arithmetic throughout: GQA repetition via `repeat_interleave` on the
/// head axis (not `candle_transformers::utils::repeat_kv`), causal masking
/// via an additive `-inf` mask built by hand, and softmax via manual
/// `exp(x - max) / sum` (not the fused `candle_nn::ops::softmax_last_dim`).
/// `QMatMul` (quantized matmul) is the one primitive kept as-is — a
/// quantized dot product isn't something worth hand-rolling; ADR-0002
/// already settled that we work on top of `candle-core`'s quantized types.
pub struct Attention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rotary: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_size: usize,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        q_proj: QMatMul,
        k_proj: QMatMul,
        v_proj: QMatMul,
        o_proj: QMatMul,
        q_norm: RmsNorm,
        k_norm: RmsNorm,
        rotary: RotaryEmbedding,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rotary,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_size: num_heads * head_dim,
        }
    }

    /// Repeat each of the `num_kv_heads` head-groups `n_rep` times along the
    /// head axis so it lines up with `num_heads` query heads — the GQA
    /// broadcast. `x` is `(b, num_kv_heads, l, head_dim)`.
    fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
        if n_rep == 1 {
            return Ok(x.clone());
        }
        let (b, kv_heads, l, d) = x.dims4()?;
        x.unsqueeze(2)?
            .broadcast_as((b, kv_heads, n_rep, l, d))?
            .reshape((b, kv_heads * n_rep, l, d))
    }

    /// Numerically stable softmax over the last dimension, computed by
    /// hand rather than via `candle_nn::ops::softmax_last_dim`.
    fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
        let last_dim = x.rank() - 1;
        let max = x.max_keepdim(last_dim)?;
        let exp = x.broadcast_sub(&max)?.exp()?;
        let sum = exp.sum_keepdim(last_dim)?;
        exp.broadcast_div(&sum)
    }

    /// `x`: `(batch, new_seq_len, hidden_size)`, already normalized (`ln1`
    /// applied by the caller) — only the *new* tokens for this step.
    /// `offset`: position of `x`'s first token in the full sequence —
    /// shifts RoPE and the causal mask. `cache`: this layer's KV cache
    /// (piece 6) — the newly computed K/V are appended to it, and
    /// attention is computed against the *full* history it returns, so a
    /// decode step (`new_seq_len == 1`) still attends every prior token.
    pub fn forward(&self, x: &Tensor, offset: usize, cache: &mut KvCache) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let n_rep = self.num_heads / self.num_kv_heads;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Per-head RMSNorm on Q and K (Qwen3-specific), flattened over
        // (batch * heads * seq_len) so RmsNorm sees plain (.., head_dim) rows.
        let q_flat = self.q_norm.forward(&q.flatten(0, 2)?)?;
        let k_flat = self.k_norm.forward(&k.flatten(0, 2)?)?;
        let q = q_flat.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.num_kv_heads, l, self.head_dim))?;

        let (q, k) = self.rotary.apply(&q, &k, offset)?;

        // Append this step's K/V to the cache and attend against the full
        // history (piece 6) — for a fresh cache and offset 0 this reduces
        // to exactly piece 4's no-cache behaviour (kv_len == l).
        let (k_full, v_full) = cache.append(&k, &v)?;
        let kv_len = k_full.dim(2)?;

        let k_full = Self::repeat_kv(&k_full, n_rep)?.contiguous()?;
        let v_full = Self::repeat_kv(&v_full, n_rep)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores =
            (q.contiguous()?.matmul(&k_full.transpose(2, 3)?.contiguous()?)? * scale)?;

        // Additive causal mask: query row i (position `offset + i`) may
        // only attend to key columns `<= offset + i` out of `kv_len`.
        let mask_vec: Vec<f32> = (0..l)
            .flat_map(|i| {
                (0..kv_len).map(move |j| {
                    if j <= i + offset {
                        0.0f32
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();
        let mask = Tensor::from_vec(mask_vec, (l, kv_len), x.device())?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let scores = scores.broadcast_add(&mask)?;

        let probs = Self::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v_full)?;
        let ctx = ctx.transpose(1, 2)?.reshape((b, l, self.hidden_size))?;
        self.o_proj.forward(&ctx)
    }
}

/// Peça 6/7 — KV Cache. Holds past K/V tensors along the sequence axis
/// (`dim=2` for our `(b, heads, seq, head_dim)` layout) and grows them on
/// each `append` call, returning the full history so far — this is what
/// lets `Attention::forward` (piece 4) be called with just the *new*
/// tokens' K/V once wired into the layer loop (piece 7), instead of
/// recomputing the whole sequence every step.
///
/// Own arithmetic: growth is a plain `Tensor::cat` along `dim`, done by
/// hand here rather than delegating to
/// `candle_nn::kv_cache::ConcatKvCache`.
pub struct KvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    dim: usize,
}

impl KvCache {
    pub fn new(dim: usize) -> Self {
        Self { k: None, v: None, dim }
    }

    /// Append the newly computed `k`/`v` slice (shape `(b, heads, new_len,
    /// head_dim)`) to whatever is cached, and return the full `(k, v)`
    /// history including the new slice.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        self.k = Some(match &self.k {
            None => k,
            Some(prev) => Tensor::cat(&[prev, &k], self.dim)?,
        });
        self.v = Some(match &self.v {
            None => v,
            Some(prev) => Tensor::cat(&[prev, &v], self.dim)?,
        });
        Ok((self.k.clone().unwrap(), self.v.clone().unwrap()))
    }

    /// Drop everything cached so the next `append` starts a fresh sequence.
    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_transformers::models::quantized_qwen3::RotaryEmbedding as OracleRotary;

    /// Piece 3 validation: our `RotaryEmbedding` against the oracle's
    /// (`candle_transformers::models::quantized_qwen3::RotaryEmbedding`,
    /// which applies the fused `candle_nn::rotary_emb::rope` kernel).
    /// Uses Qwen3-0.6B's real head_dim (128), rope_theta (1e6), and a
    /// synthetic q/k batch — RoPE only depends on shape and position, not
    /// on model weights, so no GGUF file is needed here.
    #[test]
    fn rope_matches_oracle() {
        let device = Device::Cpu;
        let head_dim = 128usize;
        let rope_theta = 1_000_000f64;
        let max_pos = 4096usize;
        let (batch, heads, seq_len) = (1usize, 4usize, 6usize);
        let offset = 3usize;

        let ours = RotaryEmbedding::new(head_dim, max_pos, rope_theta, &device).unwrap();
        let oracle =
            OracleRotary::new(DType::F32, head_dim, max_pos, rope_theta, &device).unwrap();

        let numel = batch * heads * seq_len * head_dim;
        let data: Vec<f32> = (0..numel).map(|i| (i as f32 * 0.013).sin()).collect();
        let q = Tensor::from_vec(data.clone(), (batch, heads, seq_len, head_dim), &device)
            .unwrap();
        let k = Tensor::from_vec(data, (batch, heads, seq_len, head_dim), &device).unwrap();

        let (ours_q, ours_k) = ours.apply(&q, &k, offset).unwrap();
        let (oracle_q, oracle_k) = oracle.apply(&q, &k, offset).unwrap();

        for (ours_t, oracle_t) in [(ours_q, oracle_q), (ours_k, oracle_k)] {
            let a = ours_t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let b = oracle_t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(a.len(), b.len());
            let max_diff = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(max_diff < 1e-4, "max diff {max_diff} exceeds threshold 1e-4");
        }
    }

    /// Piece 4 validation: our `Attention` (hand-rolled GQA repeat, causal
    /// mask, softmax) against an oracle-equivalent built from the same
    /// real `blk.0` GGUF weights but wired through the public fused
    /// primitives the private `AttentionWeights` struct uses internally
    /// (`candle_transformers::quantized_nn::RmsNorm`,
    /// `candle_nn::rotary_emb::rope`, `candle_transformers::utils::repeat_kv`,
    /// `candle_nn::ops::softmax_last_dim`) — `AttentionWeights` itself is
    /// private to `quantized_qwen3`, so it can't be called directly.
    ///
    /// Input `x` is real token embeddings normalized by `blk.0.attn_norm`
    /// (pieces 1+2), matching what the real forward pass feeds attention.
    #[test]
    fn attention_matches_oracle_equivalent() {
        use crate::arch::qwen::layers::TokenEmbedding;
        use candle_core::quantized::gguf_file;
        use candle_transformers::quantized_nn::RmsNorm as OracleRmsNorm;
        use candle_transformers::utils::repeat_kv as oracle_repeat_kv;
        use std::fs::File;

        const MODEL_PATH: &str = "models/qwen3-0.6b-q8_0.gguf";
        let device = Device::Cpu;

        let mut file = File::open(MODEL_PATH).expect("model file present for validation");
        let content = gguf_file::Content::read(&mut file).expect("valid gguf");
        let md = |k: &str| content.metadata.get(k).unwrap();
        let num_heads = md("qwen3.attention.head_count").to_u32().unwrap() as usize;
        let num_kv_heads = md("qwen3.attention.head_count_kv").to_u32().unwrap() as usize;
        let head_dim = md("qwen3.attention.key_length").to_u32().unwrap() as usize;
        let hidden_size = md("qwen3.embedding_length").to_u32().unwrap() as usize;
        let max_pos = md("qwen3.context_length").to_u32().unwrap() as usize;
        let eps = md("qwen3.attention.layer_norm_rms_epsilon").to_f32().unwrap() as f64;
        let rope_theta = md("qwen3.rope.freq_base").to_f32().unwrap() as f64;

        // Real input: embeddings of a short token sequence, normalized by
        // the real blk.0.attn_norm — same `x` shape the real model feeds
        // into attention.
        let embed_q = content.tensor(&mut file, "token_embd.weight", &device).unwrap();
        let embed_table = TokenEmbedding::from_qtensor(embed_q, hidden_size, &device).unwrap();
        let attn_norm_q = content.tensor(&mut file, "blk.0.attn_norm.weight", &device).unwrap();
        let attn_norm = RmsNorm::from_qtensor(attn_norm_q, eps, &device).unwrap();

        let tokens = Tensor::new(&[1u32, 100, 5000, 12, 77, 8], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let x = attn_norm.forward(&embed_table.forward(&tokens).unwrap()).unwrap();
        let offset = 0usize;

        // Load blk.0's attention weights twice — once per side — since
        // each constructor consumes its QTensor/RmsNorm.
        let load_proj = |file: &mut File, content: &gguf_file::Content, name: &str| {
            QMatMul::from_qtensor(content.tensor(file, name, &device).unwrap()).unwrap()
        };

        let ours = Attention::new(
            load_proj(&mut file, &content, "blk.0.attn_q.weight"),
            load_proj(&mut file, &content, "blk.0.attn_k.weight"),
            load_proj(&mut file, &content, "blk.0.attn_v.weight"),
            load_proj(&mut file, &content, "blk.0.attn_output.weight"),
            RmsNorm::from_qtensor(
                content.tensor(&mut file, "blk.0.attn_q_norm.weight", &device).unwrap(),
                eps,
                &device,
            )
            .unwrap(),
            RmsNorm::from_qtensor(
                content.tensor(&mut file, "blk.0.attn_k_norm.weight", &device).unwrap(),
                eps,
                &device,
            )
            .unwrap(),
            RotaryEmbedding::new(head_dim, max_pos, rope_theta, &device).unwrap(),
            num_heads,
            num_kv_heads,
            head_dim,
        );
        let mut ours_cache = KvCache::new(2);
        let ours_out = ours.forward(&x, offset, &mut ours_cache).unwrap();

        // Oracle-equivalent: same weights, same math, wired through the
        // public fused primitives instead of our hand-rolled versions.
        let oracle_q_proj = load_proj(&mut file, &content, "blk.0.attn_q.weight");
        let oracle_k_proj = load_proj(&mut file, &content, "blk.0.attn_k.weight");
        let oracle_v_proj = load_proj(&mut file, &content, "blk.0.attn_v.weight");
        let oracle_o_proj = load_proj(&mut file, &content, "blk.0.attn_output.weight");
        let oracle_q_norm = OracleRmsNorm::from_qtensor(
            content.tensor(&mut file, "blk.0.attn_q_norm.weight", &device).unwrap(),
            eps,
        )
        .unwrap();
        let oracle_k_norm = OracleRmsNorm::from_qtensor(
            content.tensor(&mut file, "blk.0.attn_k_norm.weight", &device).unwrap(),
            eps,
        )
        .unwrap();
        let oracle_rotary =
            OracleRotary::new(DType::F32, head_dim, max_pos, rope_theta, &device).unwrap();

        let (b, l, _) = x.dims3().unwrap();
        let n_rep = num_heads / num_kv_heads;
        let q = oracle_q_proj.forward(&x).unwrap();
        let k = oracle_k_proj.forward(&x).unwrap();
        let v = oracle_v_proj.forward(&x).unwrap();
        let q = q.reshape((b, l, num_heads, head_dim)).unwrap().transpose(1, 2).unwrap();
        let k = k.reshape((b, l, num_kv_heads, head_dim)).unwrap().transpose(1, 2).unwrap();
        let v = v.reshape((b, l, num_kv_heads, head_dim)).unwrap().transpose(1, 2).unwrap();
        let q_flat = candle_nn::Module::forward(&oracle_q_norm, &q.flatten(0, 2).unwrap()).unwrap();
        let k_flat = candle_nn::Module::forward(&oracle_k_norm, &k.flatten(0, 2).unwrap()).unwrap();
        let q = q_flat.reshape((b, num_heads, l, head_dim)).unwrap();
        let k = k_flat.reshape((b, num_kv_heads, l, head_dim)).unwrap();
        let (q, k) = oracle_rotary.apply(&q, &k, offset).unwrap();
        let k = oracle_repeat_kv(k, n_rep).unwrap().contiguous().unwrap();
        let v = oracle_repeat_kv(v, n_rep).unwrap().contiguous().unwrap();
        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3).unwrap()).unwrap() * scale).unwrap();
        let mask_vec: Vec<f32> = (0..l)
            .flat_map(|i| (0..l).map(move |j| if j <= i { 0.0f32 } else { f32::NEG_INFINITY }))
            .collect();
        let mask = Tensor::from_vec(mask_vec, (l, l), &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        scores = scores.broadcast_add(&mask).unwrap();
        let probs = candle_nn::ops::softmax_last_dim(&scores).unwrap();
        let ctx = probs.matmul(&v).unwrap();
        let ctx = ctx.transpose(1, 2).unwrap().reshape((b, l, num_heads * head_dim)).unwrap();
        let oracle_out = oracle_o_proj.forward(&ctx).unwrap();

        let a = ours_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let bb = oracle_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a.len(), bb.len());
        let max_diff = a
            .iter()
            .zip(bb.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "max diff {max_diff} exceeds threshold 1e-4");
    }

    /// Piece 6 validation: our `KvCache` against the oracle's
    /// `candle_nn::kv_cache::ConcatKvCache`, across a prefill step (4
    /// tokens) followed by a decode step (1 token) — the exact pattern
    /// `CandleModel::infer`'s streaming loop uses (see `model/mod.rs`).
    #[test]
    fn kv_cache_matches_oracle_concat_cache() {
        use candle_nn::kv_cache::ConcatKvCache;

        let device = Device::Cpu;
        let (b, heads, head_dim) = (1usize, 8usize, 16usize);

        let mk = |seed: f32, len: usize| {
            let numel = b * heads * len * head_dim;
            let data: Vec<f32> = (0..numel).map(|i| (seed + i as f32) * 0.01).collect();
            Tensor::from_vec(data, (b, heads, len, head_dim), &device).unwrap()
        };

        let mut ours = KvCache::new(2);
        let mut oracle = ConcatKvCache::new(2);

        // Prefill: 4 tokens.
        let (k1, v1) = (mk(0.0, 4), mk(100.0, 4));
        let (ours_k1, ours_v1) = ours.append(&k1, &v1).unwrap();
        let (oracle_k1, oracle_v1) = oracle.append(&k1, &v1).unwrap();

        // Decode: 1 new token, appended on top of the prefill.
        let (k2, v2) = (mk(50.0, 1), mk(150.0, 1));
        let (ours_k2, ours_v2) = ours.append(&k2, &v2).unwrap();
        let (oracle_k2, oracle_v2) = oracle.append(&k2, &v2).unwrap();

        for (ours_t, oracle_t) in [
            (ours_k1, oracle_k1),
            (ours_v1, oracle_v1),
            (ours_k2, oracle_k2),
            (ours_v2, oracle_v2),
        ] {
            assert_eq!(ours_t.dims(), oracle_t.dims());
            let a = ours_t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let bb = oracle_t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let max_diff = a
                .iter()
                .zip(bb.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(max_diff < 1e-4, "max diff {max_diff} exceeds threshold 1e-4");
        }

        // Reset drops history — the next append starts a fresh sequence.
        ours.reset();
        let (k3, v3) = (mk(0.0, 4), mk(100.0, 4));
        let (ours_k3, _) = ours.append(&k3, &v3).unwrap();
        assert_eq!(ours_k3.dims(), k3.dims());
    }
}
