//! CandleCLI — self-contained local inference built directly on `candle`.
//!
//! This crate exposes the public API consumed by other Rust projects
//! (AgentMesh being the first consumer). The CLI binary in `main.rs` is a
//! thin front-end over this same API.
//!
//! Design commitment: inference is **streaming**. [`CandleModel::infer`]
//! returns an iterator of tokens, never a fully materialized `String`.
//! Callers that want the whole response call `.collect::<String>()`.

pub mod arch;
pub mod cli;
pub mod model;
pub mod sampler;
pub mod session;

pub use model::CandleModel;
pub use sampler::{SamplerConfig, SamplingStrategy};

/// Crate-wide error type.
#[derive(Debug)]
pub enum Error {
    /// Filesystem / reader failure.
    Io(std::io::Error),
    /// Error surfaced by `candle` (GGUF parse, tensor op, forward pass).
    Candle(candle_core::Error),
    /// Tokenizer construction, encoding or decoding failure.
    Tokenizer(String),
    /// Higher-level model problem (wrong architecture, empty prompt, …).
    Model(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "erro de E/S: {e}"),
            Error::Candle(e) => write!(f, "erro candle: {e}"),
            Error::Tokenizer(m) => write!(f, "erro de tokenização: {m}"),
            Error::Model(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<candle_core::Error> for Error {
    fn from(e: candle_core::Error) -> Self {
        Error::Candle(e)
    }
}

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, Error>;
