# CPU: reduzir temporários e paralelizar inferência preservando SGD

Rascunho local para revisão. Este arquivo não é uma Issue publicada nem comprova abertura anterior à alteração do código.

## Diagnóstico

O legado cria vetores e copia argumentos durante forward/backward, armazena linhas de pesos separadas e calcula um gradiente dos pixels que o chamador descarta. O perfil inicial de 51 épocas registrou 9503779205 instruções Callgrind. Esse contador não mede tempo nem cache físico.

## Hipótese e alteração

Parâmetros contíguos e workspace fixo devem reduzir trabalho repetido. O gradiente de entrada descartado pode ser retirado sem modificar a atualização dos pesos. A inferência admite distribuir imagens com pesos somente para leitura e temporários privados; o SGD por amostra permanece sequencial.

## Método e correção

Intel Core i5-1240P, WSL Ubuntu 22.04 e GCC 11. Compilação C++17, O3, OpenMP e `-ffp-contract=off`. Dados oficiais: 3823 amostras de treino e 1797 de teste; 51 épocas, taxa 0.02, inicialização 1337 por camada. Foram cinco repetições, após aquecimento e com ordem aleatória fixa, de legado, sequencial e OpenMP 1/2/4/8: 30 execuções validadas.

As 51 perdas e todos os 17970 escores coincidiram exatamente, assim como previsões e matriz de confusão. Foram verificados passos SGD contra as classes originais e um gradiente por diferenças finitas.

## Resultado

| Versão | Mediana treino (s) | Mediana inferência (s, 100 passagens) |
| --- | ---: | ---: |
| Legado | 0.456937 | 0.126996 |
| Sequencial otimizada | 0.220282 | 0.135349 |
| OpenMP, quatro threads na inferência | 0.222440 | 0.050758 |

Ganho no treino sequencial de 2.07× e na inferência OpenMP de 2.50× sobre o legado. A inferência sequencial não melhorou; não se deve afirmar ganho em toda etapa. Os tempos de treino OpenMP variam por ruído, pois não houve paralelização do treino.

Um perfil separado, com três épocas e simulação de cache, registrou Ir de 994096117 para 631953618 (redução aproximada de 36.4%). Referências de dados caíram de 369289072 para 251184179; misses D1 simulados de 225854 para 148338. Os perfis incluem leitura e preparação do programa instrumentado. Não comparar diretamente esses totais com o perfil inicial de 51 épocas.

## Discussão

A redução de instruções é compatível com a retirada de alocações, cópias e contas descartadas. As taxas D1 simuladas foram baixas; os dados não demonstram que cache misses eram o gargalo dominante. O benefício deve ser discutido como redução de trabalho e movimentação evitável, sem atribuir todo o ganho ao layout.

Quatro threads venceram oito na mediana. Overhead, virtualização e carga do computador são hipóteses, não causas isoladas comprovadas por contadores. A ordem de SGD impõe uma parte sequencial: o ganho de inferência não acelera essa parte do treino.

## Reproduzir e consultar

`make cpu test` compila e valida. `python3 scripts/run_suite.py --out results/cluster/cpu-novo --repetitions 5 --infer-repeats 100` coleta uma nova bateria CPU, em uma pasta ainda não usada.

Evidências no repositório: `results/local/cpu-suite/manifest.json`, `measurements.csv`, JSONs e validações por execução; `results/local/cpu-profiles/` contém os perfis. A discussão completa está em `docs/ANALISE_CPU.md`. Marco local: `v0.2-cpu`.

Antes de publicar esta discussão, inserir os links reais da tag e das evidências na branch publicada. Não há número de Issue ou PR a referenciar nesta preparação local.
