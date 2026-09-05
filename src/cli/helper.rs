//! `rustyline` helper.
//!
//! Its only job is to color the prompt through `highlight_prompt`. That
//! path is used for rendering only — rustyline measures the prompt width
//! from the raw string passed to `readline` (see [`renderer::prompt_label`]),
//! so keeping the ANSI codes out of that string and adding them here is what
//! keeps the cursor aligned, especially on the Windows tty backend which
//! does not strip escape sequences when computing widths.
//!
//! Completion, hinting and validation are left as the trait defaults
//! (no-ops). Implemented by hand rather than via `#[derive]` to avoid
//! pulling rustyline's `derive` feature.

use std::borrow::Cow;

use rustyline::completion::Completer;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::Validator;
use rustyline::Helper;

use super::renderer;

pub struct PromptHelper;

impl Completer for PromptHelper {
    type Candidate = String;
}

impl Hinter for PromptHelper {
    type Hint = String;
}

impl Validator for PromptHelper {}

impl Highlighter for PromptHelper {
    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(
        &'s self,
        prompt: &'p str,
        _default: bool,
    ) -> Cow<'b, str> {
        Cow::Owned(renderer::styled_prompt(prompt))
    }
}

impl Helper for PromptHelper {}
