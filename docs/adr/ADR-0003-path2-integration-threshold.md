# ADR-0003 — Path 2 end-to-end integration test: tolerance threshold

- Status: **Accepted**
- Date: 2026-09-15
- Scope: Escopo 3 (Path 2), piece 7/7 — the 28-layer loop

## Context

Pieces 1–6 (embeddings, RMSNorm, RoPE, GQA attention, SwiGLU MLP, KV cache)
each validated against the oracle (`candle_transformers::models::quantized_qwen3`)
at a **1e-4** absolute-difference threshold, in isolation, with real weights
from `qwen3-0.6b-q8_0.gguf`.

Piece 7 wires all six into `Model` (`arch/qwen/layers.rs`) — the full
28-layer forward pass — and validates it end-to-end against
`ModelWeights::forward` on the same file, across a prefill step and a decode
step. At 1e-4 this failed: max diff **0.2177** on the final logits.

## Root cause

`qwen3-0.6b-q8_0.gguf` has no `general.dtype` metadata key. Oracle code
(`ModelWeights::from_gguf`):

```rust
let dtype = match gg.metadata().get("general.dtype") {
    Some(v) => match v.to_u32() { Ok(0) => DType::F32, Ok(1) => DType::F16, _ => DType::F16 },
    None => DType::F16,
};
```

Missing the key falls through to `DType::F16`. That `dtype` is used to build
the oracle's `RotaryEmbedding` cos/sin tables — so the oracle computes RoPE
in **F16** for this file. (It would also size the additive causal mask in
that dtype, but the mask is skipped entirely on the CPU code path the oracle
takes for `batch == 1`.)

Our `RotaryEmbedding` (piece 3) always computes in **F32**. Piece 3's
isolated test didn't catch this because it explicitly constructed the
oracle with `OracleRotary::new(DType::F32, ...)`, sidestepping the file's
real (missing) `general.dtype` metadata. The full-model integration test
uses the oracle's real, self-derived dtype — F16 — so the gap surfaces
there, compounding over 28 layers into a 0.2177 max diff on final logits.

This is a precision difference in the oracle's own fallback behavior, not a
logic error in Path 2: our implementation is arguably *more* correct here,
since it never drops to F16.

## Options considered

### Option 1 — Relax the threshold for this one test, add an argmax check

Keep pieces 1–6 at 1e-4 (unchanged — they don't touch this dtype fallback in
isolation). Raise the tolerance for `model_matches_oracle_end_to_end` alone
to a value that accounts for the F16-vs-F32 RoPE table gap, empirically
0.24, and document why. Add a second, independent check: the argmax
(most-likely next token) must match between our logits and the oracle's,
regardless of the absolute-difference threshold — this catches a real logic
bug (wrong token chosen) even if it wouldn't trip the relaxed magnitude
threshold.

- **Pros:** doesn't touch the six already-validated pieces; documents the
  real cause instead of guessing; the argmax check keeps the test
  meaningful as a regression guard for actual bugs, not just numeric noise.
- **Cons:** the threshold number (0.24) is empirical, not derived from
  first principles — could mask a small additional bug alongside the known
  precision gap.

### Option 2 — Deliberately downgrade our RoPE tables to F16 to match the oracle

Make our `RotaryEmbedding` compute in F16 when the oracle would, so both
sides lose precision identically and the diff shrinks back toward 1e-4.

- **Rejected.** This makes Path 2 *worse* on purpose, just to agree with an
  oracle fallback that only exists because this particular GGUF file is
  missing a metadata key. The whole point of Path 2 is to own and
  understand the math — deliberately reintroducing a precision loss we
  don't need, to chase a narrower diff number, inverts that goal for no
  real benefit.

### Option 3 — Compare only by argmax, drop the absolute-difference check

Only assert the most-likely token matches; skip magnitude comparison
entirely for the end-to-end test.

- **Rejected.** Too permissive on its own: argmax can survive a real bug
  that shifts several logits by a large, wrong amount without changing
  which one is largest (e.g., a scaling error that happens not to flip the
  top token for this particular prompt). Piece 7 is the one place we
  actually catch a 28-layer wiring bug — losing the magnitude check
  entirely removes most of that safety net.

## Decision

**Option 1.** For `model_matches_oracle_end_to_end` only:

- Absolute-difference threshold: **0.24** (was 1e-4), with the root cause
  (oracle's F16 RoPE fallback from missing `general.dtype`) documented in
  the test itself.
- Additional, independent **argmax check**: the top logit's token id must
  be identical between our `Model` and the oracle, on both the prefill and
  the decode step. A mismatch here fails the test regardless of the
  magnitude threshold — it means a real bug, not a precision artifact.

Pieces 1–6's individual validation tests are untouched — they stay at
**1e-4**, since none of them exercise the model's real (missing)
`general.dtype` fallback in isolation.

## Consequences

- `model_matches_oracle_end_to_end` in `src/arch/qwen/layers.rs` carries a
  doc comment explaining the F16/F32 RoPE gap, so a future reader doesn't
  mistake the relaxed threshold for carelessness.
- If a future GGUF file *does* carry `general.dtype = 0` (F32), the oracle
  and our implementation would agree far more closely on that file, and the
  0.24 threshold would simply not be exercised as tightly — no code change
  needed either way.
- This is scoped to the one integration test. It does not relax anything
  about how Path 2 itself computes RoPE, nor does it change the public
  `infer()` contract.
