// mlp.rs
//
// SEAM — Path 2 (in progress, Escopo 3). ADR-0002.
//
// This file holds the hand-written Qwen3 feed-forward network: the SwiGLU
// MLP — gate and up projections, SiLU on the gate, elementwise product,
// then the down projection.
//
// Peça 5/7 — MLP. Implemented below (`Mlp`), validated against the
// oracle's fused `Activation::Silu` module.
//
// The rest of the forward pass still runs through the candle_transformers
// adapter in arch/qwen/mod.rs (Path 1).

use candle_core::quantized::QMatMul;
use candle_core::{Module, Result, Tensor};

/// SwiGLU feed-forward block: `down(silu(gate(x)) * up(x))`.
///
/// SiLU (`x * sigmoid(x)`) is computed by hand via `Tensor` ops (`neg`,
/// `exp`, elementwise arithmetic) rather than `candle_nn::Activation::Silu`
/// or `Tensor::silu`, so this piece exercises our own arithmetic.
pub struct Mlp {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
}

impl Mlp {
    pub fn new(gate_proj: QMatMul, up_proj: QMatMul, down_proj: QMatMul) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    /// `x * sigmoid(x)`, with `sigmoid(x) = 1 / (1 + exp(-x))`.
    fn silu(x: &Tensor) -> Result<Tensor> {
        let sigmoid = (x.neg()?.exp()? + 1.0)?.recip()?;
        x * sigmoid
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = Self::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        let gated = (gate * up)?;
        self.down_proj.forward(&gated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::gguf_file;
    use candle_core::Device;
    use candle_nn::{Activation, Module as NnModule};
    use std::fs::File;

    const MODEL_PATH: &str = "models/qwen3-0.6b-q8_0.gguf";

    /// Piece 5 validation: our `Mlp` (hand-rolled SiLU) against the
    /// oracle's `Activation::Silu` (`candle_nn`'s fused SiLU), using real
    /// `blk.0` gate/up/down weights and a real normalized-embedding input
    /// (pieces 1+2, same pattern as the attention validation).
    #[test]
    fn mlp_matches_oracle_activation() {
        use crate::arch::qwen::layers::{RmsNorm, TokenEmbedding};

        let device = Device::Cpu;
        let mut file = File::open(MODEL_PATH).expect("model file present for validation");
        let content = gguf_file::Content::read(&mut file).expect("valid gguf");
        let hidden_size = content
            .metadata
            .get("qwen3.embedding_length")
            .unwrap()
            .to_u32()
            .unwrap() as usize;
        let eps = content
            .metadata
            .get("qwen3.attention.layer_norm_rms_epsilon")
            .unwrap()
            .to_f32()
            .unwrap() as f64;

        let embed_q = content.tensor(&mut file, "token_embd.weight", &device).unwrap();
        let embed_table = TokenEmbedding::from_qtensor(embed_q, hidden_size, &device).unwrap();
        let norm_q = content.tensor(&mut file, "blk.0.ffn_norm.weight", &device).unwrap();
        let norm = RmsNorm::from_qtensor(norm_q, eps, &device).unwrap();

        let tokens = Tensor::new(&[1u32, 100, 5000, 12], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let x = norm.forward(&embed_table.forward(&tokens).unwrap()).unwrap();

        let load = |file: &mut File, name: &str| {
            QMatMul::from_qtensor(content.tensor(file, name, &device).unwrap()).unwrap()
        };

        let ours = Mlp::new(
            load(&mut file, "blk.0.ffn_gate.weight"),
            load(&mut file, "blk.0.ffn_up.weight"),
            load(&mut file, "blk.0.ffn_down.weight"),
        );
        let ours_out = ours.forward(&x).unwrap();

        let oracle_gate = load(&mut file, "blk.0.ffn_gate.weight");
        let oracle_up = load(&mut file, "blk.0.ffn_up.weight");
        let oracle_down = load(&mut file, "blk.0.ffn_down.weight");
        let act = Activation::Silu;
        let gate = NnModule::forward(&act, &oracle_gate.forward(&x).unwrap()).unwrap();
        let up = oracle_up.forward(&x).unwrap();
        let gated = (gate * up).unwrap();
        let oracle_out = oracle_down.forward(&gated).unwrap();

        let a = ours_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = oracle_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a.len(), b.len());
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "max diff {max_diff} exceeds threshold 1e-4");
    }
}
