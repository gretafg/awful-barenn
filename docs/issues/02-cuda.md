# CUDA: preservar SGD com treino cooperativo e dados persistentes

Rascunho local para revisão. Este arquivo não é uma Issue publicada nem afirma medição de ganho ainda não realizada.

## Ponto de partida e hipótese

A versão CPU já reproduzia as contas do legado. Como esse treino pequeno leva menos de um segundo localmente, custo de contexto, cópias e lançamentos pode dominar uma migração direta.

A migração implementa treino e inferência em GPU. Compara-se, como hipótese futura, um lançamento por amostra (`cuda-naive`), por época (`cuda-global`) e por todo o treino com pesos em shared (`cuda-shared`). Os dois últimos usam inferência cooperativa com pesos reutilizados por grupo de imagens. As imagens de treino continuam sequenciais para preservar o SGD.

## Resultado real disponível

Foi executado somente `cuda-shared`, bloco 128, em 23/11/2026: 51 épocas, 3823 amostras e uma passagem de inferência de 1797 imagens. A GPU identificada na inspeção imediatamente anterior foi uma RTX A1000 de 8 GB, driver 570.124.06. O nó exato da execução não foi gravado. Compilação nvcc 12.8.93, C++17, O3, `--fmad=false`, alvo `sm_75` com PTX, e OpenMP no host.

| Região | Tempo |
| --- | ---: |
| Preparação, incluindo contexto, alocações e H2D | 1.711880743 s |
| Treino observado pelo host | 0.768943237 s |
| Treino, eventos CUDA | 768.950012 ms |
| Inferência observada pelo host | 0.000064663 s |
| Inferência, eventos CUDA | 0.060416 ms |
| H2D, já incluído na preparação | 2.432448 ms |
| D2H | 0.034912 ms |

1717/1797 acertos (95,55%). Perda de teste de 0.04319660732456998. Diferença máxima zero em perdas por época, todos os escores e parâmetros finais em relação às referências locais. Previsões e matriz de confusão coincidiram. Evidência: `results/cluster/minimo/cuda-shared.json` e `RELATO.md`.

## Discussão

Essa execução comprova funcionamento de uma configuração, não ganho estatístico. Não há medição dos backends inicial/global na mesma GPU para comparar; CPU local usou outra máquina e cem passagens na inferência. Não calcular speedup cruzando esses tempos.

O setup foi maior que o treino, reforçando a necessidade de separar o custo de iniciar a GPU e de reportar o tempo da aplicação. H2D já está no setup e não deve ser somado duas vezes. A soma de regiões não inclui leitura, inicialização CPU e exportação, portanto não é uma observação do tempo total do processo.

Shared reduz acessos explícitos dos parâmetros à memória global durante o treino e manter as épocas no kernel reduz lançamentos. A comparação entre essas hipóteses ainda não foi medida. O bloco único facilita a sincronização do SGD, mas limita o uso de múltiplos SMs. Não há métricas reais de ocupação ou conflitos de bancos para atribuir um gargalo específico.

## Pendências e reprodução

Faltam repetições, outros backends e tamanhos de bloco, Compute Sanitizer e perfis Nsight Systems/Compute. Nenhum dado de profiler GPU foi inventado para preencher essa lacuna.

No nó de acesso: `module load cuda/12.8.1` e `bash scripts/run_franky_minimo.sh` fazem uma nova execução da configuração otimizada. Para a bateria completa opcional, `bash scripts/submit_franky.sh`; ela é mais demorada e não foi iniciada nesta preparação.

Marco local: `v0.3-cuda-funcional`. Antes de publicar, inserir links reais de tag/evidências; manter explícito que a Issue não existia antes da implementação.
