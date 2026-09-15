// System prompts used by CandleCLI.
// Add new prompts here as constants — never scatter them across modules.

/// Default system prompt injected at the start of every conversation.
pub const SYSTEM_DEFAULT: &str = "\
Você é um assistente útil e direto. \
Responda apenas com base no que foi perguntado. \
Se não souber a resposta, diga que não sabe — nunca invente informações.";
