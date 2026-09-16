# CandleCLI

> 🇧🇷 [Versão em português](README.pt-BR.md)

Self-contained local LLM inference in Rust, built directly on raw
[`candle`](https://github.com/huggingface/candle). No external runtime — no
Ollama, no Python, no llama.cpp. Just a single binary.

## Two ways to use it

- **CLI** — open the terminal and land straight in a fluid chat.
- **Crate** — add `candlecli` as a dependency and call inference directly
  from your Rust code. [AgentMesh](https://github.com/) is the first
  consumer.

## Status

**Escopo 1 — done.** Local Qwen3 inference from a GGUF file, streamed token
by token, exposed both as a fluid terminal chat and as a crate API. Forward
pass: adapter over `candle_transformers`' quantized Qwen3 (ADR-0002, Path 1).
Sampler: hand-written, pure Rust. CPU only.

**Escopo 2 — done.** Persistent configuration at `~/.candlecli/` (created and
explained on first run), models registered under a short name, a default
model that auto-loads on startup, and a tokens/s figure after every reply.
Every conversation also opens with a system prompt — centralized, along with
any future ones, in [`src/prompts.rs`](src/prompts.rs). The RNG seed is
random per session by default (`/seed <value>` pins it for reproducible
output, `/seed random` clears it again).

**Escopo 3 — done.** The Qwen3 forward pass is now hand-written on
`candle-core`/`candle-nn` primitives (Path 2, ADR-0002) — embeddings,
RMSNorm, RoPE, grouped-query attention, SwiGLU MLP and the KV cache are all
our own code, validated piece by piece against `candle_transformers`'
quantized Qwen3 as a logit oracle (ADR-0003 documents the one relaxed
tolerance, in the end-to-end integration test only). `candle_transformers`
remains a dependency, but only as that test-time oracle — the CLI no
longer calls into it at runtime. Current throughput is **~5.3 tok/s** on
the Escopo 1 hand-written path, versus ~18 tok/s on the old Path 1 adapter
(which used `candle_transformers`' fused, CPU-optimized attention kernels).
Correctness was Escopo 3's goal, not performance — closing that gap
(fused/optimized kernels for our own attention and MLP) is expected to be a
future scope.

## Crate API

```rust
use candlecli::{CandleModel, SamplerConfig};

let model = CandleModel::load("qwen3-0.6b.gguf")?;

let config = SamplerConfig {
    temperature: 0.7,
    top_p: 0.9,
    ..Default::default()
};

for token in model.infer("prompt", &config)? {
    print!("{}", token?);
}
```

Inference is streaming: `infer` returns an iterator of tokens, never a
finished `String`. Call `.collect::<String>()` if you want the whole reply.

## CLI commands

| Command | Effect |
| --- | --- |
| `/model <name\|path>` | Load a registered model or a GGUF file path |
| `/model register <name> <path>` | Register a GGUF file under a short name |
| `/model default <name>` | Set the model that auto-loads on startup |
| `/model list` | List registered models |
| `/temperature <value>` | Set sampling temperature |
| `/top-p <value>` | Set top-p |
| `/top-k <value>` | Set top-k |
| `/seed <value>` | Pin the RNG seed (reproducible output) |
| `/seed random` | Unpin it (fresh seed every session) |
| `/reset` | Clear conversation history |
| `/help` | List commands |
| `/quit`, `/exit` | Leave |

## `~/.candlecli/`

Created automatically on first run:

```
~/.candlecli/
  config.toml   # sampling defaults, default model, registered models
  models/       # recommended (not required) place for registered GGUF files
  memory/       # reserved for a future memory system — empty for now
```

`config.toml` is plain TOML and safe to hand-edit:

```toml
[model]
default = "qwen3"

[sampling]
temperature = 0.8
top_p = 0.95
top_k = 40
strategy = "temperature"

[behavior]
max_new_tokens = 512

[models]
qwen3 = "~/.candlecli/models/Qwen3-0.6B-Q8_0.gguf"
```

## Design decisions

- **Architecture**: Qwen3 family (0.6B / 1.7B quantized GGUF first). Llama,
  Phi and Gemma come later as additional modules.
- **Model format**: GGUF only — vocabulary and metadata are embedded.
- **Tokenization**: `candle`'s native tokenizer; no custom BPE.
- **Sampling**: hand-written in pure Rust (not `candle`'s sampler). Greedy,
  temperature, top-p and top-k ship in Escopo 1.
- **Forward pass**: hand-written on `candle-core`/`candle-nn` primitives —
  embeddings, RMSNorm, RoPE, GQA attention, SwiGLU MLP, KV cache
  (`docs/adr/ADR-0002-qwen3-inference.md`, Path 2, Escopo 3).
- **CLI**: a plain chat loop — no menus, panels or full-screen TUI. Built on
  `crossterm` + `rustyline` (`docs/adr/ADR-0001-cli-library.md`).

Technical decisions with more than one reasonable option are recorded as
numbered ADRs under [`docs/adr/`](docs/adr/).

## Development

Build, run, download a model, and project layout: see
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

```bash
cargo run   # dev build; deps are optimized so inference is usable
```

## License

TBD.
