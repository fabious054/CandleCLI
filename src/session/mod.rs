//! Session layer.
//!
//! Owns conversational state for one chat: the ordered message history.
//! `/reset` clears it. It also assembles the model prompt by rendering the
//! history in Qwen3's ChatML format — the bare crate API stays
//! template-agnostic, so this wrapping lives here rather than in `model`.
//!
//! [`Session::render_prompt`] opens every prompt with the system turn from
//! [`crate::prompts::SYSTEM_DEFAULT`] — the single place prompt text lives
//! (see that module).

/// Role of a single message in the history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

impl Role {
    fn tag(self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

/// One entry in the conversation history.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

/// Mutable state of an ongoing chat session.
#[derive(Default)]
pub struct Session {
    history: Vec<Message>,
}

impl Session {
    /// Start an empty session.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a message to the history.
    pub fn push(&mut self, role: Role, content: String) {
        self.history.push(Message { role, content });
    }

    /// Clear the conversation history (`/reset`).
    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Render the full ChatML prompt for a new user turn: the system
    /// prompt, every past message, then `new_user`, then the open
    /// assistant tag the model continues from.
    pub fn render_prompt(&self, new_user: &str) -> String {
        let mut out = format!(
            "<|im_start|>system\n{}<|im_end|>\n",
            crate::prompts::SYSTEM_DEFAULT
        );
        for m in &self.history {
            out.push_str(&format!(
                "<|im_start|>{}\n{}<|im_end|>\n",
                m.role.tag(),
                m.content
            ));
        }
        out.push_str(&format!(
            "<|im_start|>user\n{new_user}<|im_end|>\n<|im_start|>assistant\n"
        ));
        out
    }
}
