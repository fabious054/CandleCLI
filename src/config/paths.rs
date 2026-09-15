//! `~/.candlecli` — canonical paths and directory setup.
//!
//! Escopo 2: only what is actually used — `models/` (registered GGUF
//! files, by convention) and `memory/` (reserved, empty, for a future
//! memory system). No `history/`, `logs/`, or other speculative folders.

use std::path::{Path, PathBuf};

use crate::{Error, Result};

/// Canonical locations under `~/.candlecli`.
pub struct Paths {
    /// `~/.candlecli`
    pub root: PathBuf,
    /// `~/.candlecli/models` — recommended convention, not required; a
    /// registered model can point anywhere on disk.
    pub models_dir: PathBuf,
    /// `~/.candlecli/memory` — reserved for a future memory system.
    pub memory_dir: PathBuf,
    /// `~/.candlecli/config.toml`
    pub config_file: PathBuf,
}

impl Paths {
    /// Locate `~/.candlecli` under the user's home directory. Does not
    /// touch the filesystem.
    pub fn discover() -> Result<Self> {
        let home = home_dir().ok_or_else(|| {
            Error::Model(
                "não foi possível localizar o diretório home (variáveis HOME / USERPROFILE ausentes)"
                    .into(),
            )
        })?;
        let root = home.join(".candlecli");
        Ok(Self {
            models_dir: root.join("models"),
            memory_dir: root.join("memory"),
            config_file: root.join("config.toml"),
            root,
        })
    }

    /// `true` when `~/.candlecli` does not exist yet — first run.
    pub fn is_first_run(&self) -> bool {
        !self.root.is_dir()
    }

    /// Create `models/` and `memory/` (and, as their ancestor, `root`).
    /// Idempotent — safe to call on every startup.
    pub fn ensure_layout(&self) -> Result<()> {
        std::fs::create_dir_all(&self.models_dir)?;
        std::fs::create_dir_all(&self.memory_dir)?;
        Ok(())
    }
}

/// Expand a leading `~` (as `~` or `~/…`) against the home directory.
/// Paths without one pass through unchanged — an absolute or relative path
/// is exactly as valid as `~/.candlecli/models/...` for a registered model.
pub fn expand(path_str: &str) -> PathBuf {
    if path_str == "~" {
        if let Some(home) = home_dir() {
            return home;
        }
    } else if let Some(rest) = path_str.strip_prefix("~/").or_else(|| path_str.strip_prefix("~\\")) {
        if let Some(home) = home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(path_str)
}

/// Render a path with the home directory collapsed to `~`, for display
/// only (e.g. the first-run setup log). Falls back to the plain path.
pub fn display(path: &Path) -> String {
    if let Some(home) = home_dir() {
        if let Ok(rel) = path.strip_prefix(&home) {
            let rel = rel.to_string_lossy().replace('\\', "/");
            return if rel.is_empty() {
                "~".to_string()
            } else {
                format!("~/{rel}")
            };
        }
    }
    path.display().to_string()
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}
