// attention.rs
//
// SEAM — Path 2 (future escopo). Empty by design (ADR-0002).
//
// This file will hold the hand-written Qwen3 multi-head attention:
// Q/K/V projections, rotary position embedding (RoPE) applied to Q and K,
// grouped-query attention (KV heads repeated to match Q heads), the KV
// cache read/append, scaled dot-product attention, and the output
// projection — built on candle-nn primitives (QMatMul, rotary_emb, softmax).
//
// For now the forward pass is handled by the candle_transformers adapter in
// arch/qwen/mod.rs (Path 1). When Path 2 lands, this file replaces part of
// that adapter and the candle_transformers Qwen3 model becomes a logit
// oracle for validation.
