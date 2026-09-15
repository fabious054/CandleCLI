//! Persistent configuration: `~/.candlecli/config.toml`.
//!
//! [`CandleConfig`] round-trips the TOML file: sampling defaults, the
//! generation budget, registered models (`[models]`, name -> path string,
//! `~` expanded lazily by [`paths::expand`] at load time — never baked into
//! the stored string), and the default model to auto-load on startup.
//!
//! [`paths`] resolves `~/.candlecli` itself and creates its layout.

pub mod paths;

use std::collections::BTreeMap;
use std::fs;

use serde::{Deserialize, Serialize};

use crate::sampler::{SamplerConfig, SamplingStrategy};
use crate::{Error, Result};

use paths::Paths;

/// `[model]` — which registered model (if any) loads automatically.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelSection {
    pub default: Option<String>,
}

/// `[sampling]` — mirrors the knobs in [`SamplerConfig`].
///
/// Stored as `f64` (TOML's native float), not `f32`: widening `0.8_f32` to
/// `f64` for serialization prints as `0.800000011920929` — harmless but
/// ugly in a hand-edited file. The `f32` <-> `f64` cast happens only at the
/// [`SamplerConfig`] boundary.
///
/// `seed` is absent from the TOML by default (`None`): a fresh one is
/// drawn at CLI startup and held for the session, so a repeated prompt
/// still varies run to run without pinning every session to one sequence
/// of draws forever. `/seed <n>` sets it (reproducible); `/seed random`
/// clears it back to this default.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SamplingSection {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub strategy: SamplingStrategy,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

impl Default for SamplingSection {
    fn default() -> Self {
        let d = SamplerConfig::default();
        Self {
            temperature: round_f32(d.temperature),
            top_p: round_f32(d.top_p),
            top_k: d.top_k,
            strategy: d.strategy,
            seed: d.seed,
        }
    }
}

/// Widen `f32 -> f64` for TOML storage, rounded to 6 decimals.
///
/// A plain `as f64` widens the *bit pattern*, not the decimal value: `0.8_f32`
/// is really `0.800000011920928955078125`, and `f64`'s shortest-round-trip
/// formatter then prints all of that noise. Six decimals is far more
/// precision than a sampling knob needs and round-trips to a clean literal
/// like `0.8` or `0.95`.
fn round_f32(v: f32) -> f64 {
    ((v as f64) * 1e6).round() / 1e6
}

/// `[behavior]` — generation limits not tied to a sampling strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BehaviorSection {
    pub max_new_tokens: usize,
}

impl Default for BehaviorSection {
    fn default() -> Self {
        Self {
            max_new_tokens: SamplerConfig::default().max_new_tokens,
        }
    }
}

/// Full contents of `~/.candlecli/config.toml`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CandleConfig {
    pub model: ModelSection,
    pub sampling: SamplingSection,
    pub behavior: BehaviorSection,
    /// `[models]` — registered name -> path string, as the user typed it
    /// (may start with `~`). A `BTreeMap` keeps `config.toml` and
    /// `/model list` output in a stable, alphabetical order.
    pub models: BTreeMap<String, String>,
}

impl CandleConfig {
    /// Load `config.toml`, or create it with defaults if this is the first
    /// run. Callers should have already run [`Paths::ensure_layout`].
    pub fn load_or_init(paths: &Paths) -> Result<Self> {
        if paths.config_file.is_file() {
            let text = fs::read_to_string(&paths.config_file)?;
            toml::from_str(&text).map_err(|e| Error::Config(format!("config.toml inválido: {e}")))
        } else {
            let cfg = Self::default();
            cfg.save(paths)?;
            Ok(cfg)
        }
    }

    /// Write the current configuration to `config.toml`.
    pub fn save(&self, paths: &Paths) -> Result<()> {
        let text = toml::to_string_pretty(self)
            .map_err(|e| Error::Config(format!("falha ao serializar config.toml: {e}")))?;
        fs::write(&paths.config_file, text)?;
        Ok(())
    }

    /// Build a [`SamplerConfig`] from the persisted sections. `seed` passes
    /// through as-is — `None` if absent from `config.toml`. Resolving
    /// `None` to a random, session-held value is the CLI's job
    /// (`crate::cli::run`), not this crate-facing constructor's.
    pub fn sampler_config(&self) -> SamplerConfig {
        SamplerConfig {
            temperature: self.sampling.temperature as f32,
            top_p: self.sampling.top_p as f32,
            top_k: self.sampling.top_k,
            strategy: self.sampling.strategy,
            seed: self.sampling.seed,
            max_new_tokens: self.behavior.max_new_tokens,
        }
    }

    /// Mirror a live [`SamplerConfig`] back into the persisted sections
    /// (called whenever `/temperature`, `/top-p` or `/top-k` change it).
    ///
    /// Deliberately leaves `sampling.seed` alone: those three commands are
    /// about the sampling knobs, not the seed, and the CLI resolves
    /// `SamplerConfig.seed` to a random value every session — syncing it
    /// here would silently pin that resolved value into `config.toml` the
    /// next time the user touched an unrelated setting. `/seed` is the only
    /// thing that should ever write `sampling.seed`.
    pub fn sync_sampler(&mut self, sampler: &SamplerConfig) {
        self.sampling.temperature = round_f32(sampler.temperature);
        self.sampling.top_p = round_f32(sampler.top_p);
        self.sampling.top_k = sampler.top_k;
        self.sampling.strategy = sampler.strategy;
        self.behavior.max_new_tokens = sampler.max_new_tokens;
    }
}
