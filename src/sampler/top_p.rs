//! Top-p (nucleus) sampling: sort tokens by probability, accumulate until
//! the running sum reaches `p`, keep that prefix, renormalize, then sample.
//!
//! Probabilities come from a temperature-scaled softmax. `p >= 1.0` keeps
//! the whole distribution.

use std::cmp::Ordering;

use rand::rngs::StdRng;

use super::{rng_from, sample_multinomial, softmax_scaled, Sampler};

/// Nucleus sampler.
pub struct TopP {
    p: f32,
    temperature: f32,
    rng: StdRng,
}

impl TopP {
    pub fn new(p: f32, temperature: f32, seed: u64) -> Self {
        Self {
            p,
            temperature,
            rng: rng_from(seed),
        }
    }
}

impl Sampler for TopP {
    fn sample(&mut self, logits: &[f32]) -> u32 {
        let probs = softmax_scaled(logits, self.temperature);

        let mut idx: Vec<usize> = (0..probs.len()).collect();
        idx.sort_unstable_by(|&a, &b| {
            probs[b].partial_cmp(&probs[a]).unwrap_or(Ordering::Equal)
        });

        let mut cutoff = idx.len();
        let mut acc = 0.0;
        for (rank, &i) in idx.iter().enumerate() {
            acc += probs[i];
            if acc >= self.p {
                cutoff = rank + 1;
                break;
            }
        }
        idx.truncate(cutoff);

        let kept: Vec<f32> = idx.iter().map(|&i| probs[i]).collect();
        let sum: f32 = kept.iter().sum();
        let kept: Vec<f32> = if sum > 0.0 {
            kept.iter().map(|p| p / sum).collect()
        } else {
            kept
        };

        let choice = sample_multinomial(&kept, &mut self.rng) as usize;
        idx[choice] as u32
    }
}
