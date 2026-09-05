//! Parsing and execution of `/` commands.
//!
//! Escopo 1 command set:
//!
//! - `/model <path>`       — load a GGUF file
//! - `/temperature <f32>`  — set sampling temperature
//! - `/top-p <f32>`        — set top-p
//! - `/top-k <usize>`      — set top-k
//! - `/reset`              — clear conversation history
//! - `/help`               — list available commands
//! - `/quit` | `/exit`     — leave

/// A parsed chat command.
#[derive(Debug, PartialEq)]
pub enum Command {
    Model(String),
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
        "/model" if !arg.is_empty() => Parsed::Command(Command::Model(arg.to_string())),
        "/model" => Parsed::Invalid("uso: /model <caminho>".into()),

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

/// One-line help text for every command (`/help`).
pub const HELP_TEXT: &str = "\
/model <caminho>       carrega um arquivo GGUF
/temperature <valor>   define a temperatura de sampling
/top-p <valor>         define o top-p
/top-k <valor>         define o top-k
/reset                 limpa o histórico da conversa
/help                  mostra esta ajuda
/quit, /exit           encerra";
