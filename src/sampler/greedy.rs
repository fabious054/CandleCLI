//! Greedy sampling: always pick the highest-logit token (argmax).
//!
//! Deterministic; ignores temperature, top-p and top-k.

use super::{argmax, Sampler};

/// Argmax sampler.
pub struct Greedy;

impl Sampler for Greedy {
    fn sample(&mut self, logits: &[f32]) -> u32 {
        argmax(logits)
    }
}
