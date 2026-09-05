# ADR-0001 — Interactive CLI library

- Status: **Accepted**
- Date: 2026-09-05
- Scope: Escopo 1

## Context

CandleCLI's interface is a single fluid chat in the terminal: the user opens
the binary and is immediately at a prompt. There are no menus, panels, or a
full-screen TUI. Requirements for Escopo 1:

- Read a line of user input with a visible prompt.
- Support in-line editing, history (up/down), and Ctrl-C / Ctrl-D handling.
- Print streamed tokens as they are generated, on the same line, flushing
  immediately — interleaved with the input loop.
- Small dependency footprint; the binary must stay self-contained.
- Cross-platform (Linux, macOS, Windows).

## Options

### 1. `crossterm` (+ `rustyline` or hand-rolled line editing)

- Low-level cross-platform terminal control (raw mode, key events, cursor,
  colors). No TUI framework imposed.
- Streaming output is trivial: just write to stdout and flush.
- Line editing / history is **not** included — either add `rustyline`
  (dedicated readline-style crate, brings history + editing) or write a
  minimal editor over `crossterm` key events.
- Best fit for "just a chat loop, nothing more". Matches the "no complex TUI"
  decision directly.

### 2. `ratatui` (+ `crossterm` backend)

- Immediate-mode TUI framework: widgets, layout, full-screen buffer.
- Gives a polished scrollback/input split if we ever want one.
- Heavier: pulls the full widget/layout stack, needs a render loop and an
  alternate screen. Streaming tokens means redrawing a buffer each tick.
- Over-scoped for Escopo 1 and arguably against the "no panels, no TUI"
  decision. Could be revisited in a later escopo if the chat grows a
  status area.

### 3. `inquire`

- High-level interactive **prompts** (select, confirm, text, autocomplete).
- Great for wizards / one-shot questionnaires; not built for a persistent
  chat loop with concurrent streaming output.
- Would fight the use case. Not recommended.

## Recommendation

**Option 1: `crossterm`**, paired with `rustyline` for line editing and
history (or a small hand-rolled editor if we want to avoid the extra dep).

Rationale: smallest surface that covers the requirement, keeps the binary
lean, and is the honest match for "the user opens it and is just in a chat".
`ratatui` stays available for a future escopo if the interface gains
structure; `inquire` is the wrong shape.

## Decision

**`crossterm` + `rustyline`.** No hand-rolled editor.

- `rustyline` owns input: inline editing, history, Ctrl-C / Ctrl-D.
- `crossterm` owns output: colored styling, token-by-token streaming with
  immediate flush.

The two act at distinct moments — `rustyline`'s raw mode during a read,
`crossterm` writes while printing a reply — so they do not conflict.

`Cargo.toml` gains `crossterm = "0.27"` and `rustyline = "14"`.

### Escopo 1 visual identity

Visual identity from the start — for clarity while testing, not decoration.
All of it lives in `cli/renderer.rs`; colors and screen layout can evolve
without touching inference or sampling.

- User prompt → highlighted color (green / cyan).
- Model reply → white / light grey, streamed token by token.
- `/` command feedback → yellow (`✓ Modelo carregado`, …).
- Errors → red.
- Opening header → name and version, understated.

## Consequences

- `Cargo.toml` gains `crossterm` (and possibly `rustyline`).
- `cli/renderer.rs` writes through `crossterm` / stdout; `cli/mod.rs` owns
  the read-eval loop.
- No alternate screen, no global render loop — output is append-only.
