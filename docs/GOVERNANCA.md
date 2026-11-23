# Histórico local e pendências da entrega

O fork remoto continua sem as alterações desta preparação. Foi feito um novo clone em `awful-barenn`, importando a preparação local existente sem alterar a pasta anterior `aps2`. A autorização atual permite commits locais e proíbe push. Não foram abertas ou modificadas Issues e PRs.

## Evolução preservada e organização atual

Os commits `7b59ae8` (referência do legado) e `6843878` (hipótese de CPU) já existiam antes da instrução anterior de pedir autorização para commits. A tag anotada `v0.1-baseline` foi preservada. Não houve publicação desses marcos.

Na preparação atual, o trabalho foi organizado em blocos locais:

| Branch | Conteúdo | Marco |
| --- | --- | --- |
| `perf/cpu-buffers` | Parâmetros contíguos, buffers fixos, inferência OpenMP e evidências CPU | `v0.2-cpu` |
| `feat/cuda-sgd` | Treino e inferência CUDA, teste de lógica, execução real preservada e lançadores | `v0.3-cuda-funcional` |
| `docs/entrega-local` | Consolidação do relatório, rascunhos e verificação da cópia local | Sem tag de desempenho |

As branches são integradas à `main` por merges locais. Eles não são Pull Requests publicados. A separação atual organiza o trabalho já realizado; não pretende demonstrar que Issues foram abertas antes de cada alteração nem retroagir os commits à data das medições.

## Significado das tags

- `v0.1-baseline`: referência original reproduzível com `make cpu` e `./build/legacy`.
- `v0.2-cpu`: CPU validada, com 30 medições anteriores e Callgrind. Reproduzir com `make cpu test`.
- `v0.3-cuda-funcional`: CUDA compila e a configuração compartilhada teve execução real completa validada. A tag é funcional; não afirma ganho empírico entre os backends CUDA.

As tags são anotadas e permanecem locais. Os comandos de consulta são `git show v0.2-cpu` e `git show v0.3-cuda-funcional`. Confira cada marco em uma cópia limpa ou preserve seu trabalho antes de trocar de versão.

## O que ainda falta para cumprir a rubrica

1. Revisar e publicar as discussões em Issues, com links reais para os dados e para as tags. Os [rascunhos CPU](issues/01-cpu.md) e [CUDA](issues/02-cuda.md) são arquivos locais, não Issues existentes.
2. Informar honestamente que os registros locais de hipótese não cumprem literalmente a exigência de abrir Issue antes da implementação. Não inventar cronologia, números de Issues ou discussões anteriores.
3. Executar os outros backends na GPU, medições repetidas comparáveis, verificadores e perfis Nsight, se houver tempo e autorização para essa coleta. O pedido anterior foi de execução mínima; não foi iniciada uma bateria nova nesta preparação.
4. Publicar os commits, branches e tags somente após autorização. O repositório usa `master` no remoto e a cópia final está organizada em `main` local; não houve alteração da branch padrão do GitHub.
5. Substituir as referências locais por links reais de Issues/PRs e conferir a branch de entrega antes da submissão do link no Blackboard.

A rubrica prevê nota zero para ausência de governança ou discussão empírica nas Issues. Código local validado, sozinho, não conclui esses requisitos. Nenhum documento deste clone afirma que a publicação já foi feita.
