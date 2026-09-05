//! Architecture layer.
//!
//! Defines the [`Architecture`] trait — the runtime contract a transformer
//! implementation exposes to the model layer: a forward pass over token ids
//! that returns next-token logits and maintains a KV cache, plus a way to
//! drop that cache between independent sequences.
//!
//! Construction is architecture-specific and lives on the concrete type
//! (see [`qwen::Qwen::from_gguf`]), not on the trait.
//!
//! Qwen3 is the only architecture in scope for Escopo 1. Llama, Phi and
//! Gemma will be added later as sibling modules under `arch/`.

pub mod qwen;

use crate::Result;

/// Runtime contract for a concrete transformer architecture.
pub trait Architecture {
    /// Forward pass: map `tokens` (starting at sequence position `offset`)
    /// to next-token logits over the full vocabulary, appending to the
    /// internal KV cache.
    fn forward(&mut self, tokens: &[u32], offset: usize) -> Result<Vec<f32>>;

    /// Drop the KV cache so the next [`forward`](Self::forward) starts a
    /// fresh sequence.
    fn clear_kv_cache(&mut self);
}
