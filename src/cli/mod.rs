//! CLI layer: the interactive chat.
//!
//! No menus, no panels, no complex TUI. The user opens the binary and is
//! immediately in a chat prompt; all configuration happens through `/`
//! commands (see [`commands`]). On-screen output goes through
//! [`renderer`].
//!
//! Per ADR-0001: `rustyline` owns input (inline editing, history,
//! Ctrl-C / Ctrl-D); `crossterm` owns styled output. They act at distinct
//! moments and do not conflict.

pub mod commands;
pub mod helper;
pub mod renderer;

use rustyline::error::ReadlineError;
use rustyline::history::FileHistory;
use rustyline::Editor;

use helper::PromptHelper;

use crate::model::CandleModel;
use crate::sampler::{SamplerConfig, SamplingStrategy};
use crate::session::{Role, Session};

use commands::{Command, Parsed};

/// Mutable state of the chat front-end for one run.
struct ChatState {
    /// The loaded model, once `/model` succeeds.
    model: Option<CandleModel>,
    /// Active sampling configuration, mutated by `/temperature`, `/top-p`,
    /// `/top-k`.
    sampler: SamplerConfig,
    /// Conversation history / accumulated context.
    session: Session,
}

impl ChatState {
    fn new() -> Self {
        Self {
            model: None,
            sampler: SamplerConfig::default(),
            session: Session::new(),
        }
    }
}

/// Run the interactive chat loop until the user exits.
pub fn run() {
    renderer::header();

    let mut editor: Editor<PromptHelper, FileHistory> = match Editor::new() {
        Ok(e) => e,
        Err(e) => {
            renderer::error(&format!("não foi possível iniciar o editor de linha: {e}"));
            return;
        }
    };
    editor.set_helper(Some(PromptHelper));

    let mut state = ChatState::new();
    let prompt = renderer::prompt_label();

    loop {
        match editor.readline(&prompt) {
            Ok(line) => {
                if line.trim().is_empty() {
                    continue;
                }
                let _ = editor.add_history_entry(line.as_str());
                if handle_line(&line, &mut state) == Flow::Quit {
                    break;
                }
            }
            // Ctrl-C: abandon the current line, keep going.
            Err(ReadlineError::Interrupted) => continue,
            // Ctrl-D: leave.
            Err(ReadlineError::Eof) => break,
            Err(e) => {
                renderer::error(&format!("erro de leitura: {e}"));
                break;
            }
        }
    }
}

#[derive(PartialEq)]
enum Flow {
    Continue,
    Quit,
}

/// Dispatch a single input line: run a `/` command or send chat to the model.
fn handle_line(line: &str, state: &mut ChatState) -> Flow {
    match commands::parse(line) {
        Parsed::Chat => {
            reply(line.trim(), state);
            Flow::Continue
        }
        Parsed::Invalid(msg) => {
            renderer::error(&msg);
            Flow::Continue
        }
        Parsed::Command(cmd) => run_command(cmd, state),
    }
}

fn run_command(cmd: Command, state: &mut ChatState) -> Flow {
    match cmd {
        Command::Model(path) => match CandleModel::load(&path) {
            Ok(model) => {
                state.model = Some(model);
                renderer::info(&format!("modelo carregado: {path}"));
            }
            Err(e) => renderer::error(&format!("falha ao carregar modelo: {e}")),
        },
        Command::Temperature(v) => {
            state.sampler.temperature = v;
            state.sampler.strategy = if v <= 0.0 {
                SamplingStrategy::Greedy
            } else {
                SamplingStrategy::Temperature
            };
            renderer::info(&format!("temperatura {v} — estratégia {:?}", state.sampler.strategy));
        }
        Command::TopP(v) => {
            state.sampler.top_p = v;
            state.sampler.strategy = SamplingStrategy::TopP;
            renderer::info(&format!("top-p {v} — estratégia TopP"));
        }
        Command::TopK(v) => {
            state.sampler.top_k = v;
            state.sampler.strategy = SamplingStrategy::TopK;
            renderer::info(&format!("top-k {v} — estratégia TopK"));
        }
        Command::Reset => {
            state.session.reset();
            if let Some(model) = state.model.as_ref() {
                model.reset_cache();
            }
            renderer::info("histórico limpo");
        }
        Command::Help => renderer::info(commands::HELP_TEXT),
        Command::Quit => return Flow::Quit,
    }
    Flow::Continue
}

/// Stream a model reply for `user_msg` to the screen, token by token, then
/// record both turns in the session history.
fn reply(user_msg: &str, state: &mut ChatState) {
    let Some(model) = state.model.as_ref() else {
        renderer::error("nenhum modelo carregado — use /model <caminho>");
        return;
    };

    let prompt = state.session.render_prompt(user_msg);
    match model.infer(&prompt, &state.sampler) {
        Ok(stream) => {
            // Qwen3's <think>…</think> block is kept in the prompt (quality)
            // but filtered out of what the user sees.
            let mut filter = renderer::ThinkFilter::new();
            let mut answer = String::new();
            for token in stream {
                match token {
                    Ok(text) => {
                        if let Some(visible) = filter.feed(&text) {
                            answer.push_str(&visible);
                            renderer::token(&visible);
                        }
                    }
                    Err(e) => {
                        renderer::error(&format!("erro durante a geração: {e}"));
                        break;
                    }
                }
            }
            if let Some(rest) = filter.flush() {
                answer.push_str(&rest);
                renderer::token(&rest);
            }
            renderer::end_reply();
            state.session.push(Role::User, user_msg.to_string());
            state.session.push(Role::Assistant, answer);
        }
        Err(e) => renderer::error(&format!("falha ao iniciar a inferência: {e}")),
    }
}
