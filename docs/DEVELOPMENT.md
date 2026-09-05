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

**3. Inside the chat**

```
/model models/Qwen3-0.6B-Q8_0.gguf   # load the model (~6-7s on first run)
/help                                  # list all available commands
```

Then type any prompt and press Enter.

**Available commands**

| Command | Description |
|---|---|
| `/model <path>` | Load a GGUF model file |
| `/temperature <value>` | Set sampling temperature (0 = greedy) |
| `/top-p <value>` | Set top-p sampling threshold |
| `/top-k <value>` | Set top-k sampling limit |
| `/reset` | Clear conversation history and KV cache |
| `/help` | List all commands |
| `/exit` or `/quit` | Exit CandleCLI |

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
    model/
      mod.rs            # CandleModel: load, infer, reset_cache
      gguf.rs           # GGUF file reader and architecture validation
    arch/
      mod.rs            # Architecture trait
      qwen/
        mod.rs          # Qwen3 adapter over candle_transformers (Path 1)
        attention.rs    # SEAM — reserved for Path 2 (future escopo)
        mlp.rs          # SEAM — reserved for Path 2 (future escopo)
        layers.rs       # SEAM — reserved for Path 2 (future escopo)
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
