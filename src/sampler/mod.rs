//! Sampling layer.
//!
//! Own implementation in pure Rust — the `candle` sampler is deliberately
//! not used (closed decision). Four strategies, each in its own file:
//!
//! - [`greedy`]      — argmax
//! - [`temperature`] — temperature-scaled softmax sampling
//! - [`top_k`]       — top-k truncation, then sample
//! - [`top_p`]       — nucleus sampling
//!
//! [`SamplerConfig`] carries the knobs and picks the strategy;
//! [`build`] turns it into a boxed [`Sampler`].

pub mod greedy;
pub mod temperature;
pub mod top_k;
pub mod top_p;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Which sampling strategy [`build`] instantiates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingStrategy {
    /// Deterministic argmax.
    Greedy,
    /// Temperature-scaled softmax sampling.
    Temperature,
    /// Keep the `top_k` highest logits, then sample.
    TopK,
    /// Keep the smallest prefix reaching cumulative probability `top_p`.
    TopP,
}

/// Configuration for token sampling.
///
/// `Default` is a general-purpose chat preset. Which knob matters depends on
/// `strategy`: `Temperature` uses `temperature`, `TopK` uses `top_k` (plus
/// `temperature`), `TopP` uses `top_p` (plus `temperature`), `Greedy` uses
/// none.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// Softmax temperature. `<= 0.0` collapses to greedy.
    pub temperature: f32,
    /// Nucleus threshold in `(0.0, 1.0]`. `>= 1.0` keeps the full vocab.
    pub top_p: f32,
    /// Top-k cutoff. `0` (or `>= vocab`) disables truncation.
    pub top_k: usize,
    /// Active strategy.
    pub strategy: SamplingStrategy,
    /// RNG seed, for reproducible sampling.
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 40,
            strategy: SamplingStrategy::Temperature,
            seed: 42,
        }
    }
}

/// Common contract for a sampling strategy: raw logits over the vocabulary
/// in, chosen next-token id out.
pub trait Sampler {
    fn sample(&mut self, logits: &[f32]) -> u32;
}

/// Instantiate the strategy selected by `config`.
pub fn build(config: &SamplerConfig) -> Box<dyn Sampler> {
    match config.strategy {
        SamplingStrategy::Greedy => Box::new(greedy::Greedy),
        SamplingStrategy::Temperature => {
            Box::new(temperature::Temperature::new(config.temperature, config.seed))
        }
        SamplingStrategy::TopK => Box::new(top_k::TopK::new(
            config.top_k,
            config.temperature,
            config.seed,
        )),
        SamplingStrategy::TopP => Box::new(top_p::TopP::new(
            config.top_p,
            config.temperature,
            config.seed,
        )),
    }
}

// --- shared helpers -------------------------------------------------------

/// Index of the largest logit.
pub(crate) fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// Temperature-scaled softmax. `temp <= 0.0` is treated as `1.0`; callers
/// that want greedy handle it before calling here.
pub(crate) fn softmax_scaled(logits: &[f32], temp: f32) -> Vec<f32> {
    let t = if temp <= 0.0 { 1.0 } else { temp };
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&l| ((l - max) / t).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    }
    probs
}

/// Draw an index from a probability distribution (assumed to sum to ~1).
pub(crate) fn sample_multinomial(probs: &[f32], rng: &mut StdRng) -> u32 {
    let r: f32 = rng.gen::<f32>();
    let mut acc = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if r < acc {
            return i as u32;
        }
    }
    (probs.len().saturating_sub(1)) as u32
}

/// Seed a deterministic RNG.
pub(crate) fn rng_from(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_largest() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3, -1.0]), 1);
    }

    #[test]
    fn softmax_sums_to_one_and_orders() {
        let p = softmax_scaled(&[1.0, 2.0, 3.0], 1.0);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(p[2] > p[1] && p[1] > p[0]);
    }

    #[test]
    fn greedy_is_deterministic_argmax() {
        let logits = [0.2, 5.0, 1.0, 4.9];
        let mut s = greedy::Greedy;
        assert_eq!(s.sample(&logits), 1);
        assert_eq!(s.sample(&logits), 1);
    }

    #[test]
    fn zero_temperature_collapses_to_argmax() {
        let logits = [0.2, 0.1, 9.0, 0.3];
        let mut s = temperature::Temperature::new(0.0, 42);
        assert_eq!(s.sample(&logits), 2);
    }

    #[test]
    fn top_k_one_always_picks_the_max() {
        let logits = [1.0, 0.0, 3.0, 2.0];
        let mut s = top_k::TopK::new(1, 1.0, 7);
        for _ in 0..20 {
            assert_eq!(s.sample(&logits), 2);
        }
    }

    #[test]
    fn top_p_tiny_threshold_picks_the_mode() {
        let logits = [0.0, 10.0, 0.0, 0.0];
        let mut s = top_p::TopP::new(0.01, 1.0, 7);
        for _ in 0..20 {
            assert_eq!(s.sample(&logits), 1);
        }
    }

    #[test]
    fn same_seed_same_sequence() {
        let logits = [1.0, 1.1, 0.9, 1.05, 0.95];
        let mut a = temperature::Temperature::new(1.0, 123);
        let mut b = temperature::Temperature::new(1.0, 123);
        for _ in 0..10 {
            assert_eq!(a.sample(&logits), b.sample(&logits));
        }
    }
}
