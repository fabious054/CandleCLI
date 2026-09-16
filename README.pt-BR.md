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
como API de crate. Forward pass: adapter sobre o Qwen3 quantizado do
`candle_transformers` (ADR-0002, Path 1). Sampler: escrito à mão, Rust puro.
Somente CPU.

**Escopo 2 — concluído.** Configuração persistente em `~/.candlecli/`
(criada e explicada na primeira execução), modelos registrados com nome
curto, modelo padrão que carrega automaticamente, e tokens/s exibido após
cada resposta. Toda conversa também abre com um prompt de sistema —
centralizado, junto de futuros prompts, em
[`src/prompts.rs`](src/prompts.rs). A seed do RNG é aleatória por sessão
por padrão (`/seed <valor>` fixa para saída reprodutível, `/seed random`
remove a fixação).

**Escopo 3 — concluído.** O forward pass do Qwen3 agora é escrito à mão
sobre primitivos do `candle-core`/`candle-nn` (Path 2, ADR-0002) —
embeddings, RMSNorm, RoPE, atenção com grouped-query attention, MLP com
SwiGLU e o KV cache são código nosso, validados peça por peça contra o
Qwen3 quantizado do `candle_transformers` como oráculo de logits (o
ADR-0003 documenta a única tolerância relaxada, só no teste de integração
de ponta a ponta). O `candle_transformers` continua como dependência, mas
só como esse oráculo de teste — a CLI não chama mais nele em produção. A
taxa atual é de **~5.3 tok/s** no caminho escrito à mão, contra ~18 tok/s
no antigo adapter Path 1 (que usava os kernels de atenção fundidos e
otimizados para CPU do `candle_transformers`). Correção era o objetivo do
Escopo 3, não performance — fechar essa diferença (kernels otimizados
para nossa própria atenção e MLP) é esperado como um escopo futuro.

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

## Comandos da CLI

| Comando | Efeito |
| --- | --- |
| `/model <nome\|caminho>` | Carrega um modelo registrado ou um arquivo GGUF |
| `/model register <nome> <caminho>` | Registra um GGUF com um nome curto |
| `/model default <nome>` | Define o modelo que carrega automaticamente |
| `/model list` | Lista os modelos registrados |
| `/temperature <valor>` | Define a temperatura de sampling |
| `/top-p <valor>` | Define o top-p |
| `/top-k <valor>` | Define o top-k |
| `/seed <valor>` | Fixa a seed do RNG (saída reprodutível) |
| `/seed random` | Remove a fixação (seed aleatória por sessão) |
| `/reset` | Limpa o histórico da conversa |
| `/help` | Lista os comandos |
| `/quit`, `/exit` | Encerra |

## `~/.candlecli/`

Criado automaticamente na primeira execução:

```
~/.candlecli/
  config.toml   # padrões de sampling, modelo padrão, modelos registrados
  models/       # local recomendado (não obrigatório) para os GGUF registrados
  memory/       # reservada para um sistema de memória futuro — vazia por ora
```

`config.toml` é TOML puro, seguro de editar manualmente:

```toml
[model]
default = "qwen3"

[sampling]
temperature = 0.8
top_p = 0.95
top_k = 40
strategy = "temperature"

[behavior]
max_new_tokens = 512

[models]
qwen3 = "~/.candlecli/models/Qwen3-0.6B-Q8_0.gguf"
```

## Decisões de projeto

- **Arquitetura**: família Qwen3 (0.6B / 1.7B em GGUF quantizado primeiro).
  Llama, Phi e Gemma entram depois como módulos adicionais.
- **Formato de modelo**: GGUF apenas — vocabulário e metadados embutidos.
- **Tokenização**: tokenizador nativo do `candle`; sem BPE próprio.
- **Sampling**: escrito à mão em Rust puro (não o sampler do `candle`).
  Greedy, temperatura, top-p e top-k no Escopo 1.
- **Forward pass**: escrito à mão sobre primitivos do
  `candle-core`/`candle-nn` — embeddings, RMSNorm, RoPE, atenção GQA, MLP
  com SwiGLU, KV cache (`docs/adr/ADR-0002-qwen3-inference.md`, Path 2,
  Escopo 3).
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
