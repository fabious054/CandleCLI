# ADR-0002 — Qwen3 inference: build vs. reuse candle components

- Status: **Accepted**
- Date: 2026-09-05
- Scope: Fase de Inferência (Escopo 1)

## Context

The inference phase must load a Qwen3 GGUF file and run a streaming forward
pass. The briefing fixes two points already:

- **Sampler** — own implementation in pure Rust, not candle's. (Closed, not
  in question here.)
- **Tokenizer** — candle's native tokenizer, vocabulary from the embedded
  GGUF metadata, no custom BPE. (Closed.)

What is **not** settled is how much of the GGUF I/O and the Qwen3 transformer
we write ourselves versus take from the `candle-core` / `candle-transformers`
crates we already depend on. This determines whether `arch/qwen/attention.rs`,
`mlp.rs` and `layers.rs` hold real hand-written math or become a thin adapter
over candle — which touches the "separação máxima" module design directly.
Per project convention this fork needs an ADR before implementation.

## What candle already provides (verified in the resolved sources)

- `candle_core::quantized::gguf_file::Content` — full GGUF reader: magic
  validation, metadata key/values, tensor table, lazy quantized tensor
  loading.
- `candle_core::quantized::tokenizer::TokenizerFromGguf` — trait,
  `impl`emented for `tokenizers::Tokenizer`, builds the BPE tokenizer
  (vocab + merges + pre-tokenizer + byte-level decoder) straight from GGUF
  metadata. No `tokenizer.json`, no network. This is exactly the closed
  tokenizer decision.
- `candle_transformers::models::quantized_qwen3::ModelWeights` — complete
  quantized Qwen3: `from_gguf(content, reader, device)`, `forward(input,
  offset)`, `clear_kv_cache()`. Includes RoPE, grouped-query attention,
  SwiGLU MLP, RMSNorm, KV cache.

## Cross-cutting decisions (apply to every option below)

- **GGUF byte parsing** → use `gguf_file::Content`. Hand-rolling a GGUF
  parser is high cost / zero gain — the same reasoning the briefing used to
  reject a custom BPE. `model/gguf.rs` becomes: open file, validate magic
  (delegated), read `Content`, assert `general.architecture == "qwen3"`,
  surface a descriptive error otherwise.
- **Tokenizer** → `TokenizerFromGguf`. No option here re-implements BPE.
- **Quantized weight storage / dequant** → candle `QTensor` / `QMatMul`.
  Writing our own k-quant dequantization is out of scope for Escopo 1 under
  any option.

## Options for the Qwen3 forward pass

### Option A — Reuse `candle_transformers` Qwen3; own only the sampler

- `CandleModel` wraps `quantized_qwen3::ModelWeights`.
- `arch/qwen/mod.rs` implements our `Architecture` trait as an adapter over
  `ModelWeights` (`forward` → logits).
- `arch/qwen/{attention,mlp,layers}.rs` stay as documented placeholders (or
  are removed) until a later escopo.
- **Pros:** fastest path to the Escopo 1 validation; lowest risk; RoPE/GQA/
  quant-matmul correctness is candle's problem; matches "primeiro funciona,
  depois refatora".
- **Cons:** the deliberate `arch/qwen/` breakdown carries no real code yet;
  we don't own the transformer math; conflicts with "não consolidar
  módulos por simplicidade" in spirit (structure present, empty).

### Option B — Hand-wire Qwen3 on `candle-nn` primitives

- Use `candle_nn` / `candle_core` ops: `QMatMul` for projections,
  `candle_nn::rotary_emb` for RoPE, `RmsNorm`, `softmax`, `repeat_kv` for
  GQA. GGUF parsing and `QTensor` still from candle.
- `attention.rs`, `mlp.rs`, `layers.rs` get genuine implementations; `mod.rs`
  orchestrates and exposes `forward(tokens) -> logits`.
- **Pros:** honours the module design; team owns and understands the
  architecture; extension to Llama/Phi/Gemma later is a real reuse of these
  seams.
- **Cons:** more code, more ways to be subtly wrong (RoPE base/scaling,
  head-dim layout, KV-cache offsets, quant matmul dtype). Slower to green.
  candle examples are reference-only per the briefing, so the fiddly parts
  are on us.

### Option C — Hand-roll everything including GGUF bytes and dequant

- Rejected for the same reason the briefing rejected a custom BPE: very high
  cost, no meaningful gain, large correctness surface. Not recommended.

## Recommendation

Two defensible paths depending on how much the team wants to own now:

1. **If speed to the Escopo 1 validation is the priority — Option A**, keep
   `arch/qwen/{attention,mlp,layers}.rs` as explicit "to be hollowed out"
   seams with a tracking note, and open a follow-up escopo to migrate the
   math in-house (Option B) once the end-to-end path is proven.
2. **If owning the transformer math is a hard requirement for this phase —
   Option B**, accepting a slower, riskier implementation, with candle's
   `quantized_qwen3` kept as an oracle to diff logits against during
   development.

Recommended: **path 1 (Option A now, Option B as a named follow-up)** — it
gets all five validation checks green fastest and de-risks the sampler /
CLI-integration work, which is where the closed-decision value of this
project actually is.

## Decision

- **Path 1.** `CandleModel` wraps `quantized_qwen3::ModelWeights`;
  `arch/qwen/mod.rs` implements `Architecture` as an adapter over it. When
  Escopo 1 is green, candle's model becomes the logit oracle for a dedicated
  Path 2 escopo.
- **Seams kept.** `arch/qwen/{attention,mlp,layers}.rs` stay as empty,
  documented `SEAM — Path 2` files. Not removed, not implemented now.
- **CPU only.** `Device::Cpu` hardcoded. No CUDA/Metal wiring; GPU is a
  future opt-in via candle feature flags.
- **Dev model.** `Qwen3-0.6B-GGUF / qwen3-0.6b-q4_k_m.gguf`. Contingency:
  if throughput `< 2–3 tok/s` or quality is poor at validation, switch to
  `qwen3-1.7b-q4_k_m.gguf` — same code, different file. The earlier
  AgentMesh slowness was kalosm's generic load path (AgentMesh ADR-0004),
  not the GGUF; CandleCLI calls `ModelWeights` directly.

### Implementation notes

- `infer(&self, …)` keeps its signature. The streaming iterator holds
  `Arc<Mutex<Qwen>>` + `Arc<Tokenizer>`, so `CandleModel` stays
  `Send + Sync` and the boxed iterator stays `'static`.
- Each `infer` call is an independent sequence: it clears the KV cache on
  entry. Multi-turn context is rebuilt by the CLI from `Session` history as
  a ChatML string; the bare crate API stays template-agnostic.
- CLI ergonomics: `/temperature 0` selects `Greedy`, `/temperature >0`
  selects `Temperature`, `/top-k` selects `TopK`, `/top-p` selects `TopP`,
  so every knob is observable. Not a `/strategy` command (out of the Escopo
  1 command set).

## Follow-up

- **Path 2 escopo** — hand-write the Qwen3 forward pass across the seam
  files, diffing logits against `quantized_qwen3` as oracle.
- If AgentMesh needs `infer` to be cancellable or to expose token ids /
  logprobs, that is a separate ADR against the public API.

## Consequences

- `model/gguf.rs` depends on `candle_core::quantized::gguf_file`.
- Tokenizer construction lands in `model/mod.rs` (or a small
  `model/tokenizer.rs`) via `TokenizerFromGguf`.
- Under Option A, `arch/` is an adapter layer for now; under Option B it is
  the implementation.
- The sampler work (`sampler/`) and CLI integration are identical either
  way and can proceed in parallel once this is decided.
