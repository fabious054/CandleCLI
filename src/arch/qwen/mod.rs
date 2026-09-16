//! Qwen3 architecture — Path 2, hand-written forward pass (ADR-0002,
//! ADR-0003).
//!
//! The forward pass is [`layers::Model`] — our own embeddings, RMSNorm,
//! RoPE, GQA attention, SwiGLU MLP and KV cache, assembled across 28
//! layers (Escopo 3). This module wraps it behind
//! [`crate::arch::Architecture`] and converts between our plain `&[u32]` /
//! `Vec<f32>` interface and candle tensors.
//!
//! `candle_transformers::models::quantized_qwen3::ModelWeights` is no
//! longer used here — it remains a dependency only as the logit oracle in
//! `#[cfg(test)]` code across [`attention`], [`mlp`] and [`layers`].

pub mod attention;
pub mod layers;
pub mod mlp;

use std::io::{Read, Seek};

use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};

use crate::arch::Architecture;
use crate::Result;

/// A loaded Qwen3 model (quantized weights + KV cache).
pub struct Qwen {
    model: layers::Model,
    device: Device,
}

impl Qwen {
    /// Build from parsed GGUF contents. Consumes `content`; reads tensor
    /// data through `reader`.
    pub fn from_gguf<R: Read + Seek>(
        content: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let model = layers::Model::from_gguf(content, reader, device)?;
        Ok(Self {
            model,
            device: device.clone(),
        })
    }
}

impl Architecture for Qwen {
    fn forward(&mut self, tokens: &[u32], offset: usize) -> Result<Vec<f32>> {
        // (seq_len,) -> (1, seq_len)
        let input = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;
        // Model::forward returns last-position logits, shape (1, vocab).
        let logits = self.model.forward(&input, offset)?;
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
        Ok(logits.to_vec1::<f32>()?)
    }

    fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }
}
