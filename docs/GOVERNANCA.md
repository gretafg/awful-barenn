# Histórico e publicação

O fork [gretafg/awful-barenn](https://github.com/gretafg/awful-barenn) contém código, evidências e tags anotadas. A branch principal é `main`; o antigo `master` foi preservado. As Issues foram habilitadas e os registros publicados após a implementação, sem retroagir datas.

## Blocos e rastreabilidade

| Bloco | Branch | Tag publicada | Registro |
| --- | --- | --- | --- |
| Referência do legado | Histórico inicial preservado | [v0.1-baseline](https://github.com/gretafg/awful-barenn/tree/v0.1-baseline) | [Issue #1](https://github.com/gretafg/awful-barenn/issues/1) |
| CPU e OpenMP | `perf/cpu-buffers` | [v0.2-cpu](https://github.com/gretafg/awful-barenn/tree/v0.2-cpu) | [Issue #2](https://github.com/gretafg/awful-barenn/issues/2) |
| Migração CUDA funcional | `feat/cuda-sgd` | [v0.3-cuda-funcional](https://github.com/gretafg/awful-barenn/tree/v0.3-cuda-funcional) | [Issue #3](https://github.com/gretafg/awful-barenn/issues/3) |

Os commits iniciais `7b59ae8` e `6843878` foram preservados. A organização posterior do trabalho ocorreu em branches e merges locais integrados à `main`, que foram publicados como tais. Eles não são apresentados como Pull Requests anteriores. `docs/entrega-local` reúne a verificação da cópia antes da publicação; `docs/publicacao` atualiza o índice com os registros reais.

## Escopo dos fechamentos

- Issue #1: referência original executada e dados iniciais de tempo/perfil registrados.
- Issue #2: otimização CPU e inferência OpenMP, com comparação numérica, 30 medições e perfis Callgrind.
- Issue #3: migração CUDA funcional, compilação, teste de lógica CPU e uma execução real completa de `cuda-shared` validada contra referências. O fechamento se refere à funcionalidade desse marco.

As três estão fechadas e vinculadas às respectivas tags. Os textos em `docs/issues/` preservam cópias dos registros publicados. Não foram criados perfis, números de experimentos ou resultados que não existiam.

## Limites para avaliar a entrega

1. As Issues são retrospectivas. Sua publicação agora não cumpre literalmente a exigência de abrir uma Issue antes de alterar o código. Essa diferença permanece explícita nos registros.
2. CPU e CUDA foram integradas inicialmente por merges locais, sem PRs históricos. Não houve reescrita das datas ou das tags para simular outro fluxo.
3. Ainda faltam execução dos outros backends na GPU, repetições comparáveis, Compute Sanitizer e perfis Nsight. A tag CUDA é funcional; não afirma conclusão de otimização avançada ou ganho entre versões GPU.
4. Os tempos CPU e GPU usam ambientes e quantidades de inferência diferentes. Não é válido calcular speedup direto entre eles.
5. O repositório foi publicado, mas não houve envio ao Blackboard. A entrega pedida pelo enunciado é o link do fork.

Para reproduzir o marco CPU: `make cpu test`. Para uma execução do CUDA otimizado no Franky: `module load cuda/12.8.1` e `bash scripts/run_franky_minimo.sh`. A coleta completa é opcional e mais demorada; não foi iniciada nesta publicação.
