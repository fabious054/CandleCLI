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
//!
//! Escopo 2 adds `~/.candlecli` persistence: on startup, [`run`] discovers
//! `~/.candlecli` (`crate::config::paths`), runs first-run setup if it
//! doesn't exist yet, loads `config.toml` (`crate::config`), seeds the
//! sampler from it, and auto-loads the default model if one is set.

pub mod commands;
pub mod helper;
pub mod renderer;

use std::time::Instant;

use rustyline::error::ReadlineError;
use rustyline::history::FileHistory;
use rustyline::Editor;

use helper::PromptHelper;

use crate::config::{paths, paths::Paths, CandleConfig};
use crate::model::CandleModel;
use crate::sampler::{SamplerConfig, SamplingStrategy};
use crate::session::{Role, Session};

use commands::{Command, Parsed};

/// Mutable state of the chat front-end for one run.
struct ChatState {
    /// The loaded model, once `/model` or the default-model auto-load
    /// succeeds.
    model: Option<CandleModel>,
    /// Active sampling configuration, mutated by `/temperature`, `/top-p`,
    /// `/top-k` — and mirrored back into `config` on every change.
    sampler: SamplerConfig,
    /// Conversation history / accumulated context.
    session: Session,
    /// `~/.candlecli/config.toml`, kept in sync with `sampler` and with
    /// `/model register` / `/model default`.
    config: CandleConfig,
    /// Canonical `~/.candlecli` locations.
    paths: Paths,
}

/// Run the interactive chat loop until the user exits.
pub fn run() {
    renderer::header();

    let paths = match Paths::discover() {
        Ok(p) => p,
        Err(e) => {
            renderer::error(&format!("{e}"));
            return;
        }
    };
    let first_run = paths.is_first_run();

    if first_run {
        renderer::blank();
        renderer::plain("Primeira execução detectada.");
        renderer::plain(&format!("→ Criando {}...", paths::display(&paths.root)));
        renderer::plain(&format!(
            "→ Criando {}...",
            paths::display(&paths.models_dir)
        ));
        renderer::plain(&format!(
            "→ Criando {}...",
            paths::display(&paths.memory_dir)
        ));
        renderer::plain("→ Gerando config.toml com padrões...");
    }

    if let Err(e) = paths.ensure_layout() {
        renderer::error(&format!(
            "falha ao criar {}: {e}",
            paths::display(&paths.root)
        ));
        return;
    }

    let config = match CandleConfig::load_or_init(&paths) {
        Ok(c) => c,
        Err(e) => {
            renderer::error(&format!("falha ao carregar configuração: {e}"));
            return;
        }
    };

    if first_run {
        renderer::info(
            "Configuração inicial concluída.\n  Para começar: /model register <nome> <caminho-do-arquivo.gguf>",
        );
        renderer::blank();
    }

    let mut editor: Editor<PromptHelper, FileHistory> = match Editor::new() {
        Ok(e) => e,
        Err(e) => {
            renderer::error(&format!("não foi possível iniciar o editor de linha: {e}"));
            return;
        }
    };
    editor.set_helper(Some(PromptHelper));

    let mut state = ChatState {
        sampler: config.sampler_config(),
        model: None,
        session: Session::new(),
        config,
        paths,
    };

    if let Some(default_name) = state.config.model.default.clone() {
        load_default_model(&default_name, &mut state);
        renderer::blank();
    }

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
        Command::Model(name_or_path) => load_named_or_path(&name_or_path, state),
        Command::ModelRegister(name, path) => register_model(name, path, state),
        Command::ModelDefault(name) => set_default_model(name, state),
        Command::ModelList => list_models(state),
        Command::Temperature(v) => {
            state.sampler.temperature = v;
            state.sampler.strategy = if v <= 0.0 {
                SamplingStrategy::Greedy
            } else {
                SamplingStrategy::Temperature
            };
            persist_sampler(state);
            renderer::info(&format!(
                "temperatura {v} — estratégia {:?}",
                state.sampler.strategy
            ));
        }
        Command::TopP(v) => {
            state.sampler.top_p = v;
            state.sampler.strategy = SamplingStrategy::TopP;
            persist_sampler(state);
            renderer::info(&format!("top-p {v} — estratégia TopP"));
        }
        Command::TopK(v) => {
            state.sampler.top_k = v;
            state.sampler.strategy = SamplingStrategy::TopK;
            persist_sampler(state);
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

/// Load a model given either a registered name (looked up in
/// `config.models`) or a bare path — mirrors `CandleModel::load`'s own
/// tolerance for either.
fn load_named_or_path(name_or_path: &str, state: &mut ChatState) {
    let path_str = state
        .config
        .models
        .get(name_or_path)
        .cloned()
        .unwrap_or_else(|| name_or_path.to_string());
    let resolved = paths::expand(&path_str);

    match CandleModel::load(&resolved) {
        Ok(model) => {
            state.model = Some(model);
            renderer::info(&format!("modelo carregado: {name_or_path}"));
        }
        Err(e) => renderer::error(&format!("falha ao carregar modelo `{name_or_path}`: {e}")),
    }
}

/// Load the startup default model, with the two failure modes the briefing
/// calls out explicitly: not registered, or registered but missing on disk.
fn load_default_model(name: &str, state: &mut ChatState) {
    let Some(path_str) = state.config.models.get(name).cloned() else {
        renderer::error(&format!(
            "modelo padrão `{name}` não está registrado. Use /model register {name} <caminho>."
        ));
        return;
    };

    let resolved = paths::expand(&path_str);
    if !resolved.is_file() {
        renderer::error(&format!(
            "modelo padrão `{name}` está registrado, mas o arquivo não existe mais: {}. \
             Registre de novo com /model register {name} <caminho>.",
            resolved.display()
        ));
        return;
    }

    renderer::plain(&format!("→ Carregando modelo padrão: {name}..."));
    match CandleModel::load(&resolved) {
        Ok(model) => {
            state.model = Some(model);
            renderer::info(&format!("{name} carregado."));
        }
        Err(e) => renderer::error(&format!("falha ao carregar modelo padrão `{name}`: {e}")),
    }
}

fn register_model(name: String, path: String, state: &mut ChatState) {
    let resolved = paths::expand(&path);
    if !resolved.is_file() {
        renderer::error(&format!(
            "arquivo não encontrado: {}",
            resolved.display()
        ));
        return;
    }

    state.config.models.insert(name.clone(), path.clone());
    match state.config.save(&state.paths) {
        Ok(()) => renderer::info(&format!("modelo registrado: {name} → {path}")),
        Err(e) => renderer::error(&format!("falha ao salvar config.toml: {e}")),
    }
}

fn set_default_model(name: String, state: &mut ChatState) {
    if !state.config.models.contains_key(&name) {
        renderer::error(&format!(
            "modelo `{name}` não está registrado — use /model register {name} <caminho>"
        ));
        return;
    }

    state.config.model.default = Some(name.clone());
    match state.config.save(&state.paths) {
        Ok(()) => renderer::info(&format!(
            "modelo padrão definido: {name} (carrega automaticamente na próxima abertura)"
        )),
        Err(e) => renderer::error(&format!("falha ao salvar config.toml: {e}")),
    }
}

fn list_models(state: &ChatState) {
    if state.config.models.is_empty() {
        renderer::info("nenhum modelo registrado — use /model register <nome> <caminho>");
        return;
    }

    let mut lines = String::new();
    for (name, path) in &state.config.models {
        let is_default = state.config.model.default.as_deref() == Some(name.as_str());
        let marker = if is_default { " (padrão)" } else { "" };
        lines.push_str(&format!("{name}{marker} → {path}\n"));
    }
    renderer::info(lines.trim_end());
}

/// Mirror `state.sampler` into `state.config` and persist it — called by
/// every command that mutates the sampler, so temperature/strategy survive
/// a restart.
fn persist_sampler(state: &mut ChatState) {
    state.config.sync_sampler(&state.sampler);
    if let Err(e) = state.config.save(&state.paths) {
        renderer::error(&format!("falha ao salvar configuração: {e}"));
    }
}

/// Stream a model reply for `user_msg` to the screen, token by token, then
/// record both turns in the session history.
fn reply(user_msg: &str, state: &mut ChatState) {
    let Some(model) = state.model.as_ref() else {
        renderer::error("nenhum modelo carregado — use /model <nome|caminho>");
        return;
    };

    let prompt = state.session.render_prompt(user_msg);
    match model.infer(&prompt, &state.sampler) {
        Ok(stream) => {
            // Qwen3's <think>…</think> block is kept in the prompt (quality)
            // but filtered out of what the user sees.
            let mut filter = renderer::ThinkFilter::new();
            let mut answer = String::new();
            let mut generated = 0usize;
            let started = Instant::now();

            for token in stream {
                generated += 1;
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
            let elapsed = started.elapsed();

            if let Some(rest) = filter.flush() {
                answer.push_str(&rest);
                renderer::token(&rest);
            }
            renderer::end_reply();

            let secs = elapsed.as_secs_f64();
            if generated > 0 && secs > 0.0 {
                renderer::token_rate(generated as f64 / secs);
            }

            state.session.push(Role::User, user_msg.to_string());
            state.session.push(Role::Assistant, answer);
        }
        Err(e) => renderer::error(&format!("falha ao iniciar a inferência: {e}")),
    }
}
