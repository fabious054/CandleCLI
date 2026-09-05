# CandleCLI

> 🇬🇧 [English version](README.md)

Inferência local de LLM em Rust, autossuficiente, construída direto sobre o
[`candle`](https://github.com/huggingface/candle) cru. Sem runtime externo —
sem Ollama, sem Python, sem llama.cpp. Só um binário.

## Dois modos de uso

- **CLI** — abra o terminal e caia direto num chat fluido.
- **Crate** — adicione `candlecli` como dependência e chame inferência direto
  no seu código Rust. O AgentMesh é o primeiro consumidor.

## Estado atual

**Escopo 1 — concluído.** Inferência local do Qwen3 a partir de um arquivo
GGUF, com streaming token a token, exposta como chat fluido no terminal e
como API de crate:

1. Chat fluido no terminal (header + prompt). ✅
2. Comandos `/` para carregar modelo e configurar sampling. ✅
3. Prompt do usuário lido no chat. ✅
4. Arquivo GGUF carregado do disco (arquitetura Qwen3 validada). ✅
5. Inferência via `candle` com streaming de tokens. ✅
6. Tokens impressos conforme são gerados. ✅

Forward pass: adapter sobre o Qwen3 quantizado do `candle_transformers`
(ADR-0002, Path 1). Sampler: escrito à mão, Rust puro. Somente CPU.

## API do crate

```rust
use candlecli::{CandleModel, SamplerConfig};

let model = CandleModel::load("qwen3-0.6b.gguf")?;

let config = SamplerConfig {
    temperature: 0.7,
    top_p: 0.9,
    ..Default::default()
};

for token in model.infer("prompt", &config)? {
    print!("{}", token?);
}
```

A inferência é streaming: `infer` retorna um iterador de tokens, nunca uma
`String` pronta. Faça `.collect::<String>()` se quiser a resposta inteira.

## Comandos da CLI (Escopo 1)

| Comando | Efeito |
| --- | --- |
| `/model <caminho>` | Carrega um arquivo GGUF |
| `/temperature <valor>` | Define a temperatura de sampling |
| `/top-p <valor>` | Define o top-p |
| `/top-k <valor>` | Define o top-k |
| `/reset` | Limpa o histórico da conversa |
| `/help` | Lista os comandos |
| `/quit`, `/exit` | Encerra |

## Decisões de projeto

- **Arquitetura**: família Qwen3 (0.6B / 1.7B em GGUF quantizado primeiro).
  Llama, Phi e Gemma entram depois como módulos adicionais.
- **Formato de modelo**: GGUF apenas — vocabulário e metadados embutidos.
- **Tokenização**: tokenizador nativo do `candle`; sem BPE próprio.
- **Sampling**: escrito à mão em Rust puro (não o sampler do `candle`).
  Greedy, temperatura, top-p e top-k no Escopo 1.
- **CLI**: um loop de chat simples — sem menus, painéis ou TUI de tela
  cheia. Construída sobre `crossterm` + `rustyline`
  (`docs/adr/ADR-0001-cli-library.md`).

Decisões técnicas com mais de uma opção razoável são registradas como ADRs
numerados em [`docs/adr/`](docs/adr/).

## Desenvolvimento

Build, execução, download de um modelo e estrutura do projeto: ver
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) (em inglês).

```bash
cargo run   # build dev; as dependências vão otimizadas, inferência já é usável
```

## Licença

A definir.
