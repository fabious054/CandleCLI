// mlp.rs
//
// SEAM — Path 2 (future escopo). Empty by design (ADR-0002).
//
// This file will hold the hand-written Qwen3 feed-forward network: the
// SwiGLU MLP — gate and up projections, SiLU on the gate, elementwise
// product, then the down projection — built on candle-nn primitives.
//
// For now the forward pass is handled by the candle_transformers adapter in
// arch/qwen/mod.rs (Path 1).
