# Análise dos experimentos locais em CPU

Os resultados locais validam a implementação e permitem avaliar as primeiras hipóteses. Eles não substituem o experimento no Franky.

## Método

Foram medidas cinco execuções de cada uma das seis configurações: legado instrumentado, CPU sequencial e OpenMP com 1, 2, 4 e 8 threads. A ordem foi embaralhada com semente fixa, após aquecimento por configuração. Cada execução treinou 51 épocas com 3823 imagens e classificou 1797 imagens em 100 passagens de inferência.

O ambiente foi WSL Ubuntu 22.04 sobre Intel Core i5-1240P. Legado e CPU foram compilados com O3, símbolos de depuração e contração de ponto flutuante desabilitada. Os comandos e a descrição de ambiente estão em `results/local/cpu-suite/manifest.json`; os tempos individuais estão em `measurements.csv`.

Todas as perdas por época, escores, previsões e matrizes de confusão comparados coincidiram. A avaliação produziu 1717 acertos em 1797 imagens, como o legado. O teste adicional de aceitação exige igualdade exata entre CPU e referência.

## Resultados

O treino mediano caiu de 0.456937 s no legado para 0.220282 s na versão sequencial, um speedup de aproximadamente 2.07. A implementação remove cópias e alocações no caminho repetitivo e deixa de calcular o gradiente de entrada que o programa original descartava.

A inferência sequencial otimizada não apresentou ganho mediano nessa coleta: 0.135349 s, frente a 0.126996 s do legado. Os intervalos se sobrepõem. Portanto, os dados não sustentam afirmar que toda etapa da versão sequencial ficou mais rápida.

Com quatro threads, a inferência teve mediana de 0.050758 s, aproximadamente 2.67 vezes mais rápida que a versão sequencial otimizada e 2.50 vezes frente ao legado. O paralelismo distribui imagens independentes com pesos imutáveis e um espaço de trabalho próprio por iteração.

O backend OpenMP não paraleliza o treino SGD. A diferença entre os tempos de treino das configurações OpenMP decorre da variabilidade da execução; seria incorreto atribuí-la ao número de threads.

Oito threads não venceram quatro na mediana. A hipótese é que o custo de coordenação, a carga do computador e a execução virtualizada limitem a escala nesse lote. Não há contadores físicos nesta medição que permitam atribuir uma causa única.

## Perfil de instruções e cache simulado

Uma coleta separada com Callgrind, cache e desvios simulados usou três épocas nas duas versões:

| Métrica | Legado | CPU otimizada |
| --- | ---: | ---: |
| Instruções Ir | 994096117 | 631953618 |
| Referências a dados | 369289072 | 251184179 |
| Misses D1 simulados | 225854 | 148338 |
| Desvios | 132828073 | 69595462 |

A redução aproximada de 36.4% em Ir é compatível com a eliminação de trabalho, mas não prevê diretamente a redução do tempo. Os contadores incluem leitura e preparação dos dados, além do treino e da inferência desse programa instrumentado.

Os misses são de um modelo de cache do Valgrind, não medições dos caches físicos do processador. As taxas de misses D1 foram baixas nas duas versões; a coleta não demonstra que falta de cache era o gargalo dominante. O benefício deve ser discutido principalmente como redução de trabalho e movimentação evitável, sem atribuir todo o speedup ao achatamento dos pesos.

Os relatórios e os logs originais ficam em `results/local/cpu-profiles/`. O perfil inicial de 51 épocas do programa original está em `results/local/callgrind.legacy`; ele tem escopo diferente e não deve ser comparado diretamente aos totais de três épocas.

## Limites da conclusão

As medições ocorreram em um computador de uso geral e tiveram dispersão visível. Os scripts do cluster fixam os recursos de OpenMP e registram a afinidade para repetir a comparação em condições mais controladas.

Um kernel CUDA mais rápido não garante aplicação mais rápida. A próxima análise deve incorporar contexto, alocações, transferências, kernels e retorno, preservando o mesmo trabalho. Nenhum tempo ou speedup de GPU foi preenchido antecipadamente.
