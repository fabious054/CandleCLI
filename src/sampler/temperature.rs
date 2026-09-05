//! Temperature sampling: scale logits by `1/temperature`, softmax, then
//! draw from the resulting distribution.
//!
//! Higher temperature flattens the distribution (more random); lower
//! sharpens it. `temperature <= 0.0` falls back to argmax.

use rand::rngs::StdRng;

use super::{argmax, rng_from, sample_multinomial, softmax_scaled, Sampler};

/// Temperature-scaled softmax sampler.
pub struct Temperature {
    temperature: f32,
    rng: StdRng,
}

impl Temperature {
    pub fn new(temperature: f32, seed: u64) -> Self {
        Self {
            temperature,
            rng: rng_from(seed),
        }
    }
}

impl Sampler for Temperature {
    fn sample(&mut self, logits: &[f32]) -> u32 {
        if self.temperature <= 0.0 {
            return argmax(logits);
        }
        let probs = softmax_scaled(logits, self.temperature);
        sample_multinomial(&probs, &mut self.rng)
    }
}
