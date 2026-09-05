//! GGUF format reader.
//!
//! Responsibility: open a `.gguf` file, validate it, and hand the parsed
//! contents plus a positioned reader to `model/mod.rs`.
//!
//! Per ADR-0002 the byte-level parsing is delegated to
//! `candle_core::quantized::gguf_file` — hand-rolling it buys nothing. This
//! module adds the CandleCLI-specific checks on top: valid GGUF magic
//! (enforced by `Content::read`) and `general.architecture == "qwen3"`.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use candle_core::quantized::gguf_file;

use crate::{Error, Result};

/// A parsed GGUF file together with a reader positioned for tensor loading.
pub struct GgufFile {
    /// Parsed header: metadata key/values and the tensor table.
    pub content: gguf_file::Content,
    /// Reader over the same file, used to pull tensor data on demand.
    pub reader: BufReader<File>,
}

impl GgufFile {
    /// Open and validate a GGUF file.
    ///
    /// Fails if the file is missing, is not valid GGUF, or does not carry a
    /// Qwen3 model.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut reader = BufReader::new(File::open(path)?);

        // `Content::read` validates the GGUF magic and version.
        let content = gguf_file::Content::read(&mut reader)?;

        let arch = content
            .metadata
            .get("general.architecture")
            .ok_or_else(|| Error::Model("GGUF sem `general.architecture`".into()))?
            .to_string()
            .map_err(|_| Error::Model("`general.architecture` não é uma string".into()))?;

        if arch != "qwen3" {
            return Err(Error::Model(format!(
                "arquitetura não suportada: `{arch}` (esperado `qwen3`)"
            )));
        }

        Ok(Self { content, reader })
    }
}
