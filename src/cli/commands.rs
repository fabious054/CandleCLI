//! Parsing and execution of `/` commands.
//!
//! Command set:
//!
//! - `/model <name|path>`              — load a registered model or a GGUF path
//! - `/model register <name> <path>`   — register a GGUF file under a short name
//! - `/model default <name>`           — set the model that auto-loads on startup
//! - `/model list`                     — list registered models
//! - `/temperature <f32>`              — set sampling temperature
//! - `/top-p <f32>`                    — set top-p
//! - `/top-k <usize>`                  — set top-k
//! - `/reset`                          — clear conversation history
//! - `/help`                           — list available commands
//! - `/quit` | `/exit`                 — leave

/// A parsed chat command.
#[derive(Debug, PartialEq)]
pub enum Command {
    Model(String),
    ModelRegister(String, String),
    ModelDefault(String),
    ModelList,
    Temperature(f32),
    TopP(f32),
    TopK(usize),
    Reset,
    Help,
    Quit,
}

/// Outcome of parsing one input line.
pub enum Parsed {
    /// The line is a valid `/` command.
    Command(Command),
    /// The line starts with `/` but is not a valid command.
    Invalid(String),
    /// The line is not a command — it is chat input for the model.
    Chat,
}

const MODEL_USAGE: &str = "uso: /model <nome|caminho> | /model register <nome> <caminho> | /model default <nome> | /model list";

/// Parse one input line.
pub fn parse(line: &str) -> Parsed {
    let trimmed = line.trim();
    if !trimmed.starts_with('/') {
        return Parsed::Chat;
    }

    let mut parts = trimmed.splitn(2, char::is_whitespace);
    let name = parts.next().unwrap_or("");
    let arg = parts.next().map(str::trim).unwrap_or("");

    match name {
        "/model" => parse_model(arg),

        "/temperature" => match arg.parse::<f32>() {
            Ok(v) => Parsed::Command(Command::Temperature(v)),
            Err(_) => Parsed::Invalid("uso: /temperature <valor>".into()),
        },
        "/top-p" => match arg.parse::<f32>() {
            Ok(v) => Parsed::Command(Command::TopP(v)),
            Err(_) => Parsed::Invalid("uso: /top-p <valor>".into()),
        },
        "/top-k" => match arg.parse::<usize>() {
            Ok(v) => Parsed::Command(Command::TopK(v)),
            Err(_) => Parsed::Invalid("uso: /top-k <valor>".into()),
        },

        "/reset" => Parsed::Command(Command::Reset),
        "/help" => Parsed::Command(Command::Help),
        "/quit" | "/exit" => Parsed::Command(Command::Quit),

        other => Parsed::Invalid(format!("comando desconhecido: {other}")),
    }
}

/// Parse everything after `/model `: a subcommand (`register`, `default`,
/// `list`) or, failing that, a registered name / path to load.
fn parse_model(arg: &str) -> Parsed {
    if arg.is_empty() {
        return Parsed::Invalid(MODEL_USAGE.into());
    }

    let mut sub = arg.splitn(2, char::is_whitespace);
    let head = sub.next().unwrap_or("");
    let rest = sub.next().map(str::trim).unwrap_or("");

    match head {
        "register" => {
            let mut reg = rest.splitn(2, char::is_whitespace);
            let model_name = reg.next().unwrap_or("");
            let path = reg.next().map(str::trim).unwrap_or("");
            if model_name.is_empty() || path.is_empty() {
                Parsed::Invalid("uso: /model register <nome> <caminho>".into())
            } else {
                Parsed::Command(Command::ModelRegister(
                    model_name.to_string(),
                    path.to_string(),
                ))
            }
        }
        "default" if !rest.is_empty() => {
            Parsed::Command(Command::ModelDefault(rest.to_string()))
        }
        "default" => Parsed::Invalid("uso: /model default <nome>".into()),
        "list" if rest.is_empty() => Parsed::Command(Command::ModelList),
        // Not a subcommand — the whole argument is a registered name or a
        // path (which may itself be a bare word like "list" with no
        // registered model of that name; the loader will say so).
        _ => Parsed::Command(Command::Model(arg.to_string())),
    }
}

/// One-line help text for every command (`/help`).
pub const HELP_TEXT: &str = "\
/model <nome|caminho>          carrega um modelo registrado ou um arquivo GGUF
/model register <nome> <caminho>   registra um GGUF com um nome curto
/model default <nome>          define o modelo que carrega automaticamente
/model list                    lista os modelos registrados
/temperature <valor>           define a temperatura de sampling
/top-p <valor>                 define o top-p
/top-k <valor>                 define o top-k
/reset                         limpa o histórico da conversa
/help                          mostra esta ajuda
/quit, /exit                   encerra";
