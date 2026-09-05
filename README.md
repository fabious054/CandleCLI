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
by token, exposed both as a fluid terminal chat and as a crate API:

1. Fluid chat in the terminal (header + prompt). ✅
2. `/` commands to load a model and configure sampling. ✅
3. User prompt taken from the chat. ✅
4. GGUF file loaded from disk (Qwen3 architecture validated). ✅
5. Inference through `candle` with token streaming. ✅
6. Tokens printed as they are generated. ✅

Forward pass: adapter over `candle_transformers`' quantized Qwen3
(ADR-0002, Path 1). Sampler: hand-written, pure Rust. CPU only.

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

## CLI commands (Escopo 1)

| Command | Effect |
| --- | --- |
| `/model <path>` | Load a GGUF file |
| `/temperature <value>` | Set sampling temperature |
| `/top-p <value>` | Set top-p |
| `/top-k <value>` | Set top-k |
| `/reset` | Clear conversation history |
| `/help` | List commands |
| `/quit`, `/exit` | Leave |

## Design decisions

- **Architecture**: Qwen3 family (0.6B / 1.7B quantized GGUF first). Llama,
  Phi and Gemma come later as additional modules.
- **Model format**: GGUF only — vocabulary and metadata are embedded.
- **Tokenization**: `candle`'s native tokenizer; no custom BPE.
- **Sampling**: hand-written in pure Rust (not `candle`'s sampler). Greedy,
  temperature, top-p and top-k ship in Escopo 1.
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
