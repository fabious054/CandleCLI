# CandleCLI

Ferramenta Rust autossuficiente de inferência local, construída direto sobre
o `candle` cru. Sem runtime externo — sem Ollama, sem Python, sem llama.cpp.
Binário auto-contido.

## Dois modos de uso

- **CLI** — o usuário abre o terminal e cai direto num chat fluido.
- **Crate** — qualquer projeto Rust importa o `candlecli` como dependência e
  chama inferência direto no código. O AgentMesh é o primeiro consumidor.

## Estado atual — Escopo 2 (persistência + tok/s)

Escopo 1 fechado — relatório em `../scope-reports/escopo-1-candlecli.md`.

Escopo 2 adiciona `~/.candlecli/` (config, modelos registrados, memória
reservada vazia): setup automático na primeira execução, `config.toml` em
TOML/serde persistindo sampler + modelo padrão, `/model register`,
`/model default`, `/model list`, e tokens/s exibido após cada resposta.
Módulo novo: `config/` (`mod.rs` + `paths.rs`). Também novo: `prompts.rs`
— constantes de prompt de sistema, centralizadas ali (nunca hardcoded em
outro módulo); `session::render_prompt` injeta `SYSTEM_DEFAULT` como
primeiro turno ChatML.

Forward pass do Qwen3: **Path 1** (ADR-0002) — adapter sobre
`candle_transformers::models::quantized_qwen3::ModelWeights`. Os arquivos
`arch/qwen/{attention,mlp,layers}.rs` são seams vazios para o Path 2.

Sampler: implementação própria em Rust puro, quatro estratégias, seleção via
`SamplerConfig.strategy`.

Device: `Device::Cpu` fixo. GPU é escopo futuro.

`tokenizers` **fixado em 0.23** para casar com o `candle-core` — o
`TokenizerFromGguf` do candle é implementado no `Tokenizer` daquela versão;
versão divergente não enxerga o trait.

## Decisões técnicas (fechadas — não reabrir)

- **Arquitetura**: família Qwen3, começando com 0.6B / 1.7B em GGUF
  quantizado. Llama, Phi, Gemma entram depois como módulos novos em `arch/`.
- **Formato**: GGUF exclusivamente. Vocabulário e metadados vêm embutidos no
  arquivo.
- **Tokenização**: tokenizador nativo do `candle`. Sem BPE próprio.
- **Sampling**: implementação própria em Rust puro (não usar o sampler do
  `candle`). Quatro estratégias no Escopo 1 — greedy, temperatura, top-p,
  top-k — cada uma em arquivo separado, configuradas via `SamplerConfig`
  com `Default` sensato.
- **API do crate**: streaming desde o início. `infer` retorna um iterador de
  tokens, nunca uma `String` pronta. Quem quiser o texto completo faz
  `.collect::<String>()`.
- **CLI**: chat fluido, sem menus / painéis / TUI complexa. Toda config via
  comandos `/`. Biblioteca de terminal ainda em aberto — ver
  `docs/adr/ADR-0001-cli-library.md`.

## Comandos `/`

`/model <nome|caminho>` · `/model register <nome> <caminho>` ·
`/model default <nome>` · `/model list` · `/temperature <valor>` ·
`/top-p <valor>` · `/top-k <valor>` · `/reset` · `/help` · `/quit` | `/exit`

## Estrutura de módulos

Separação máxima e intencional — cada responsabilidade no seu arquivo.
**Não consolidar módulos por simplicidade.** O projeto foi pensado pra
crescer.

```
src/
  main.rs             ponto de entrada; inicializa a CLI
  lib.rs              interface pública do crate
  prompts.rs          prompts de sistema — centralizados, nunca hardcoded
  config/
    mod.rs            CandleConfig: carregar, salvar, defaults
    paths.rs          ~/.candlecli, caminhos canônicos, criação de diretórios
  model/
    mod.rs            trait Model, orquestra carregamento
    gguf.rs           parser e loader do formato GGUF
  arch/
    mod.rs            trait Architecture
    qwen/
      mod.rs          implementação Qwen completa
      attention.rs    mecanismo de atenção
      mlp.rs          camadas MLP
      layers.rs       bloco transformer
  sampler/
    mod.rs            trait Sampler, SamplerConfig
    greedy.rs
    temperature.rs
    top_p.rs
    top_k.rs
  session/
    mod.rs            Session, histórico, contexto acumulado
  cli/
    mod.rs            loop principal do chat
    commands.rs       parsing e execução dos comandos /
    renderer.rs       renderização visual na tela
```

## Convenções

- **CLAUDE.md** → português.
- **Comentários de código e ADRs** → inglês.
- **README.md** → inglês, com link para o `README.pt-BR.md`.
- **README.pt-BR.md** → português.
- **docs/DEVELOPMENT.md** → inglês. Como buildar/rodar, baixar modelo,
  estrutura de pastas, e como o projeto é desenvolvido.
- **ADRs**: toda decisão técnica com mais de uma opção razoável vira um ADR
  numerado em `docs/adr/`, no formato `ADR-XXXX-titulo-curto.md`. O Claude
  Code propõe, a decisão sai na conversa de produto, só então implementa.

## Não fazer

- Não usar kalosm, llama.cpp, Ollama ou qualquer abstração externa.
- Não consolidar módulos por simplicidade — a separação é intencional.
- Não implementar lógica de inferência antes da estrutura estar fechada.
- Não escolher a biblioteca de CLI sem passar pelo ADR.
- Não reabrir as decisões técnicas fechadas acima.
- Não criar `~/.candlecli/history/`, `logs/` ou outras pastas especulativas
  — só o que é usado agora (`models/`, `memory/` vazia).
- Não implementar o sistema de memória — `memory/` existe mas fica vazia.
- Não modificar a interface pública de `infer()`.
- Não iniciar o Path 2 (forward pass do Qwen3 em `candle-nn` puro).
