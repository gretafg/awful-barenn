# Hipótese de otimização em CPU

Registro local aberto antes da alteração da implementação. A publicação como Issue ainda depende do fork.

## Diagnóstico

O Callgrind do legado compilado com O3 registrou 9.503.779.205 instruções (Ir), das quais 85,81% foram atribuídas a `main` após inlining. O relatório completo fica em `results/local/callgrind.legacy`. Ir representa instruções, não segundos ou cache misses.

A inspeção do código identifica construção recorrente de vetores em forward e backward, cópias dos argumentos e cálculo do gradiente da entrada de 64 valores que é descartado. O perfil por linha será usado junto dessa inspeção; não se assume que todas as alocações sejam o gargalo dominante.

## Hipótese

Usar dimensões fixas, pesos contíguos e buffers reutilizados, eliminando apenas o gradiente final não consumido, deve reduzir instruções e tempo mantendo SGD por amostra. Separar parâmetros e buffers também permite inferência multicore sem condições de corrida.

## Comparação planejada

Legado e CPU otimizada usarão os mesmos dados, épocas, taxa, sementes e flags de ponto flutuante. Serão comparadas todas as perdas por época e todos os escores do teste, além de previsões e tempos. A inferência OpenMP será comparada em lote com 1, 2, 4 e 8 threads. Tempo de treino e inferência serão apresentados separadamente.
