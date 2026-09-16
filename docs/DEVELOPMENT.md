# Development

## Requirements

- [Rust](https://rustup.rs/) stable toolchain (1.78+)
- A Qwen3 GGUF model file — see [Running locally](#running-locally)
- No Python, no Ollama, no external runtime required

Windows note: the linker warning `LNK4098: defaultlib 'LIBCMT' conflicts` is
expected — it comes from a native math kernel dependency, not from CandleCLI's
own code. It can be safely ignored.

---

## Running locally

**1. Download a Qwen3 GGUF model**

Create a `models/` folder in the repo root and download a quantized Qwen3 file:

```bash
# Recommended for development (~640 MB, Q8_0)
curl -L -o models/Qwen3-0.6B-Q8_0.gguf \
  https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf
```

The `models/` folder is in `.gitignore` — model files are never committed.

**2. Run in development mode**

```bash
cargo run
```

Dependencies (candle and its math kernels) are compiled with `opt-level = 3`
even in the dev profile — see `[profile.dev.package."*"]` in `Cargo.toml` —
so inference is usable without a full release build. CandleCLI's own code is
still built unoptimized and fully debuggable. A release build is still faster;
use it for anything performance-sensitive.

**3. First run**

The first time `candlecli` runs it creates `~/.candlecli/` (config,
`models/`, an empty `memory/` reserved for a future memory system) and
prints what it did. From then on that block doesn't show again — see
[`~/.candlecli/`](../README.md#candlecli) in the README for the layout and
`config.toml` format.

**4. Inside the chat**

```
/model register qwen3 models/Qwen3-0.6B-Q8_0.gguf   # register once
/model default qwen3                                 # optional: auto-load next time
/model qwen3                                          # load by name (~6-7s on first load)
/help                                                  # list all available commands
```

A bare path still works too: `/model models/Qwen3-0.6B-Q8_0.gguf`.

Then type any prompt and press Enter. A `▸ N.N tok/s` line follows each
reply.

**Available commands**

| Command | Description |
|---|---|
| `/model <name\|path>` | Load a registered model or a GGUF file path |
| `/model register <name> <path>` | Register a GGUF file under a short name |
| `/model default <name>` | Set the model that auto-loads on startup |
| `/model list` | List registered models |
| `/temperature <value>` | Set sampling temperature (0 = greedy) |
| `/top-p <value>` | Set top-p sampling threshold |
| `/top-k <value>` | Set top-k sampling limit |
| `/seed <value>` | Pin the RNG seed (reproducible output) |
| `/seed random` | Unpin it (fresh seed every session) |
| `/reset` | Clear conversation history and KV cache |
| `/help` | List all commands |
| `/exit` or `/quit` | Exit CandleCLI |

`/temperature`, `/top-p` and `/top-k` persist to `config.toml` immediately —
they survive a restart. `/seed <value>` also persists (as `sampling.seed`);
`/seed random` removes that key again. The other three sampling commands
never touch `sampling.seed` — only `/seed` writes it.

---

## Building for production

```bash
cargo build --release
```

The binary is produced at `target/release/candlecli` (or `candlecli.exe` on
Windows). It is fully self-contained — no runtime dependencies beyond the OS.
Copy the binary anywhere and point it at a GGUF file.

---

## Project structure

```
CandleCLI/
  src/
    main.rs             # entry point — initialises the CLI
    lib.rs              # public crate interface
    prompts.rs          # system prompt constants — never hardcoded elsewhere
    config/
      mod.rs            # CandleConfig: load/save config.toml, defaults
      paths.rs          # ~/.candlecli, canonical paths, directory setup
    model/
      mod.rs            # CandleModel: load, infer, reset_cache
      gguf.rs           # GGUF file reader and architecture validation
    arch/
      mod.rs            # Architecture trait
      qwen/
        mod.rs          # Qwen3 wrapper over layers::Model (Path 2, ADR-0002)
        attention.rs    # Attention (GQA), RotaryEmbedding (RoPE), KvCache
        mlp.rs          # Mlp — SwiGLU feed-forward
        layers.rs       # TokenEmbedding, RmsNorm, Layer, Model (28-layer loop)
    sampler/
      mod.rs            # SamplerConfig, SamplingStrategy, Sampler trait
      greedy.rs         # argmax
      temperature.rs    # temperature scaling + softmax
      top_k.rs          # top-k sampling
      top_p.rs          # nucleus (top-p) sampling
    session/
      mod.rs            # Session: conversation history, ChatML rendering
    cli/
      mod.rs            # chat loop (rustyline), command dispatch, reply()
      commands.rs       # Command enum, parse(), HELP_TEXT
      renderer.rs       # crossterm output, ThinkFilter (<think> suppression)
      helper.rs         # rustyline helper — colors the prompt via highlight_prompt
  docs/
    adr/                # Architecture Decision Records
    DEVELOPMENT.md      # this file
  models/               # GGUF model files — gitignored, never committed
  Cargo.toml            # deps + [profile.dev.package."*"] opt-level override
  CLAUDE.md             # permanent instructions for Claude Code sessions
  README.md             # English readme (links to README.pt-BR.md)
  README.pt-BR.md       # Portuguese readme
```

**Path 2 (Escopo 3)**: `arch/qwen/{attention,mlp,layers}.rs` hold the real
hand-written Qwen3 forward pass — no more adapter over
`candle_transformers`. `candle_transformers` is still a dependency, but only
as the logit oracle each piece's `#[cfg(test)]` code validates against
(never referenced from non-test code). Current throughput is ~5.3 tok/s,
down from ~18 tok/s on the old Path 1 adapter, since our attention/MLP
don't yet use the fused, CPU-optimized kernels `candle_transformers` ships
(flash-attention-style decode, interleaved KV cache layout). Escopo 3's
goal was correctness, not speed — see
[docs/adr/ADR-0002-qwen3-inference.md](adr/ADR-0002-qwen3-inference.md) and
[docs/adr/ADR-0003-path2-integration-threshold.md](adr/ADR-0003-path2-integration-threshold.md).

---

## How this project is developed

CandleCLI is built across two parallel conversations:

| Role | Where |
|---|---|
| Product decisions | Dedicated chat conversation (this document's origin) |
| Implementation | Claude Code (VS Code), with direct repo access |

The two sides never see each other directly. What connects them are artifacts:

| Artifact | Location | Language | Purpose |
|---|---|---|---|
| `CLAUDE.md` | repo root | Portuguese | Permanent instructions Claude Code reads at session start |
| ADRs | `docs/adr/` | English | Any technical decision with more than one reasonable option |
| Scope reports | `scope-reports/` (outside repo) | Portuguese | Progress log + closing report per scope |
| Trello board | CandleCLI board | Portuguese | Nothing gets lost between sessions |

**The basic cycle for a scope**

1. Product conversation defines decisions — each one becomes a Trello card
2. Claude Code receives a briefing with all closed decisions
3. Implementation proceeds; any new fork with two reasonable options → ADR proposed by Claude Code → brought to product conversation → decided → Claude Code implements
4. Validation checklist before any closing commit:
   - Manual live validation confirmed
   - Dev scaffolding removed or documented
   - READMEs updated to reflect real state
   - Version synced across config files
   - Test suite green
   - Closing commit + milestone tag + push
   - Scope report frozen

**Why ADRs exist**

Any technical decision with more than one reasonable path gets an ADR. This
captures not just what was decided but why — and what trade-offs were accepted.
Future contributors (and future Claude Code sessions) can read the ADR instead
of reconstructing the reasoning from code alone.

**Why scope reports live outside the repo**

Scope reports are a log of the product conversation — decisions, context,
things that were tried and discarded. They belong to the project's history but
not to its codebase. Keeping them outside the repo avoids polluting the commit
history with process documents and keeps the repo focused on what ships.
