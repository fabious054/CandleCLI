// layers.rs
//
// SEAM — Path 2 (future escopo). Empty by design (ADR-0002).
//
// This file will hold the hand-written Qwen3 building blocks that wrap
// attention.rs and mlp.rs into a full transformer stack: token embedding,
// RMSNorm, one transformer block (input norm -> attention -> residual ->
// post-attention norm -> MLP -> residual), the final norm, and the output
// projection to vocabulary logits.
//
// For now the forward pass is handled by the candle_transformers adapter in
// arch/qwen/mod.rs (Path 1).
