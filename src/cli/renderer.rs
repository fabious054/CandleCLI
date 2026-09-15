//! On-screen rendering for the chat.
//!
//! Responsibility: everything the user sees — the opening header, the input
//! prompt label, streamed assistant tokens printed as they arrive, `/`
//! command feedback, and error lines. Keeps terminal-output concerns out of
//! the chat loop and the command handlers.
//!
//! All color / layout decisions live here (ADR-0001). Nothing in `model`,
//! `arch` or `sampler` should ever print.

use std::io::{self, Write};

use crossterm::style::{Color, Print, ResetColor, SetForegroundColor};
use crossterm::{execute, queue};

/// Color of the user input prompt label.
const USER_COLOR: Color = Color::Cyan;
/// Color of streamed model output.
const MODEL_COLOR: Color = Color::Grey;
/// Color of `/` command feedback.
const INFO_COLOR: Color = Color::Yellow;
/// Color of error lines.
const ERROR_COLOR: Color = Color::Red;
/// Color of the understated opening header.
const HEADER_COLOR: Color = Color::DarkGrey;

/// Print the opening header: name and version, understated.
pub fn header() {
    let _ = execute!(
        io::stdout(),
        SetForegroundColor(HEADER_COLOR),
        Print(format!("candlecli {}\n", env!("CARGO_PKG_VERSION"))),
        ResetColor,
    );
}

/// The prompt label handed to `rustyline`, **uncolored**.
///
/// Color is applied later by [`styled_prompt`] through the editor's
/// highlighter. If the ANSI codes were in this string, rustyline would
/// count them as visible columns (its Windows tty backend does not strip
/// escapes) and misplace the cursor.
pub fn prompt_label() -> String {
    "› ".to_string()
}

/// Colorize the prompt for display. Called by the rustyline helper's
/// `highlight_prompt`, never used for width measurement.
pub fn styled_prompt(prompt: &str) -> String {
    use crossterm::style::Stylize;
    prompt.to_string().with(USER_COLOR).to_string()
}

/// Print a single streamed token, no trailing newline, flushed immediately
/// so it shows up as it is generated.
pub fn token(text: &str) {
    let mut out = io::stdout();
    let _ = queue!(
        out,
        SetForegroundColor(MODEL_COLOR),
        Print(text),
        ResetColor,
    );
    let _ = out.flush();
}

/// Terminate the current streamed reply with a newline.
pub fn end_reply() {
    let _ = execute!(io::stdout(), Print("\n"));
}

/// Print an uncolored line — first-run setup narration (`Primeira
/// execução detectada.`, `→ Criando …`) that is neither success/error
/// feedback nor model output.
pub fn plain(text: &str) {
    let _ = execute!(io::stdout(), Print(format!("{text}\n")));
}

/// Print a blank line — spacing around the first-run setup block and the
/// default-model auto-load line.
pub fn blank() {
    let _ = execute!(io::stdout(), Print("\n"));
}

/// Print the tokens/s figure for the reply that just finished. One
/// discreet line, in the header's muted color — a number, not decoration
/// (dedicated visual treatment is a future escopo).
pub fn token_rate(tokens_per_sec: f64) {
    let _ = execute!(
        io::stdout(),
        SetForegroundColor(HEADER_COLOR),
        Print(format!("▸ {tokens_per_sec:.1} tok/s\n")),
        ResetColor,
    );
}

/// Print a `/` command feedback line, e.g. `✓ Modelo carregado`.
pub fn info(text: &str) {
    let _ = execute!(
        io::stdout(),
        SetForegroundColor(INFO_COLOR),
        Print(format!("✓ {text}\n")),
        ResetColor,
    );
}

/// Print an error line.
pub fn error(text: &str) {
    let _ = execute!(
        io::stderr(),
        SetForegroundColor(ERROR_COLOR),
        Print(format!("✗ {text}\n")),
        ResetColor,
    );
}

// --- Qwen3 <think> block filter -----------------------------------------

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThinkState {
    /// Outside the block — text is printed.
    Outside,
    /// Inside `<think>…</think>` — text is consumed silently.
    InsideThink,
}

/// Stateful filter that strips Qwen3's `<think>…</think>` reasoning block
/// out of a token stream.
///
/// Qwen3 emits the block by design (kept in the ChatML template to preserve
/// answer quality); the user should only see the final answer. The tags may
/// arrive whole (they are single special tokens) or, with a different model
/// or quant, split across tokens — hence the lookahead buffer.
///
/// One instance per reply, created in `reply()` and dropped at the end. Not
/// global state.
pub struct ThinkFilter {
    state: ThinkState,
    /// Lookahead: text not yet safe to emit because it might be the start
    /// of a tag.
    buf: String,
    /// After leaving a think block, swallow the leading whitespace that
    /// precedes the answer.
    trim_leading: bool,
}

impl Default for ThinkFilter {
    fn default() -> Self {
        Self {
            state: ThinkState::Outside,
            buf: String::new(),
            trim_leading: false,
        }
    }
}

impl ThinkFilter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one streamed fragment. Returns the text that should be printed,
    /// or `None` when this fragment produced nothing visible.
    pub fn feed(&mut self, fragment: &str) -> Option<String> {
        self.buf.push_str(fragment);
        let mut out = String::new();

        loop {
            match self.state {
                ThinkState::Outside => {
                    if let Some(pos) = self.buf.find(THINK_OPEN) {
                        out.push_str(&self.buf[..pos]);
                        self.buf.drain(..pos + THINK_OPEN.len());
                        self.state = ThinkState::InsideThink;
                        continue;
                    }
                    let hold = partial_tag_suffix(&self.buf, THINK_OPEN);
                    let upto = self.buf.len() - hold;
                    out.push_str(&self.buf[..upto]);
                    self.buf.drain(..upto);
                    break;
                }
                ThinkState::InsideThink => {
                    if let Some(pos) = self.buf.find(THINK_CLOSE) {
                        self.buf.drain(..pos + THINK_CLOSE.len());
                        self.state = ThinkState::Outside;
                        self.trim_leading = true;
                        continue;
                    }
                    let hold = partial_tag_suffix(&self.buf, THINK_CLOSE);
                    let drop_upto = self.buf.len() - hold;
                    self.buf.drain(..drop_upto);
                    break;
                }
            }
        }

        self.finish_chunk(out)
    }

    /// Emit whatever is safely left once the stream ends. A dangling partial
    /// tag, or anything still inside an unclosed think block, is dropped.
    pub fn flush(&mut self) -> Option<String> {
        let leftover = std::mem::take(&mut self.buf);
        let out = if self.state == ThinkState::Outside && !looks_like_partial_tag(&leftover) {
            leftover
        } else {
            String::new()
        };
        self.finish_chunk(out)
    }

    fn finish_chunk(&mut self, out: String) -> Option<String> {
        let out = if self.trim_leading {
            let trimmed = out.trim_start();
            if !trimmed.is_empty() {
                self.trim_leading = false;
            }
            trimmed.to_string()
        } else {
            out
        };
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }
}

/// Length of the longest suffix of `buf` that is a non-empty proper prefix
/// of `tag` — the bytes we must hold back in case the rest of the tag is
/// still coming. Returns 0 if `buf` already contains the full tag or no
/// prefix at all.
fn partial_tag_suffix(buf: &str, tag: &str) -> usize {
    let max = buf.len().min(tag.len() - 1);
    for k in (1..=max).rev() {
        let start = buf.len() - k;
        if buf.is_char_boundary(start) && buf[start..] == tag[..k] {
            return k;
        }
    }
    0
}

fn looks_like_partial_tag(buf: &str) -> bool {
    partial_tag_suffix(buf, THINK_OPEN) == buf.len()
        || partial_tag_suffix(buf, THINK_CLOSE) == buf.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(fragments: &[&str]) -> String {
        let mut f = ThinkFilter::new();
        let mut s = String::new();
        for frag in fragments {
            if let Some(out) = f.feed(frag) {
                s.push_str(&out);
            }
        }
        if let Some(out) = f.flush() {
            s.push_str(&out);
        }
        s
    }

    #[test]
    fn passes_through_plain_text() {
        assert_eq!(run(&["Hello", ", ", "world"]), "Hello, world");
    }

    #[test]
    fn strips_whole_think_block() {
        assert_eq!(
            run(&["<think>", "reasoning here", "</think>", "The answer."]),
            "The answer."
        );
    }

    #[test]
    fn strips_block_with_surrounding_newlines() {
        assert_eq!(
            run(&["<think>\n", "stuff\n", "</think>\n\n", "Brasília."]),
            "Brasília."
        );
    }

    #[test]
    fn handles_tags_split_across_fragments() {
        assert_eq!(
            run(&["<", "th", "ink", ">", "hmm", "<", "/think", ">", "done"]),
            "done"
        );
    }

    #[test]
    fn keeps_text_before_the_block() {
        // Text before the block survives; the single space that led the
        // post-block fragment is folded away with the rest of the block's
        // trailing whitespace.
        assert_eq!(run(&["ok ", "<think>x</think>", " go"]), "ok go");
    }

    #[test]
    fn drops_unclosed_block() {
        assert_eq!(run(&["answer ", "<think>", "never closes"]), "answer ");
    }

    #[test]
    fn lone_angle_bracket_is_not_swallowed() {
        assert_eq!(run(&["a < b and c > d"]), "a < b and c > d");
    }
}
