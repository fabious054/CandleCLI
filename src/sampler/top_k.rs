//! Top-k sampling: keep only the `k` highest-logit tokens, softmax over
//! that subset (temperature-scaled), then sample.
//!
//! `k == 0` or `k >= vocab` degrades to plain temperature sampling.

use std::cmp::Ordering;

use rand::rngs::StdRng;

use super::{rng_from, sample_multinomial, softmax_scaled, Sampler};

/// Top-k truncation sampler.
pub struct TopK {
    k: usize,
    temperature: f32,
    rng: StdRng,
}

impl TopK {
    pub fn new(k: usize, temperature: f32, seed: u64) -> Self {
        Self {
            k,
            temperature,
            rng: rng_from(seed),
        }
    }
}

impl Sampler for TopK {
    fn sample(&mut self, logits: &[f32]) -> u32 {
        if self.k == 0 || self.k >= logits.len() {
            let probs = softmax_scaled(logits, self.temperature);
            return sample_multinomial(&probs, &mut self.rng);
        }

        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_unstable_by(|&a, &b| {
            logits[b].partial_cmp(&logits[a]).unwrap_or(Ordering::Equal)
        });
        idx.truncate(self.k);

        let kept: Vec<f32> = idx.iter().map(|&i| logits[i]).collect();
        let probs = softmax_scaled(&kept, self.temperature);
        let choice = sample_multinomial(&probs, &mut self.rng) as usize;
        idx[choice] as u32
    }
}
