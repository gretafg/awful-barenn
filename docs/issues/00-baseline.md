# Baseline: preservar o legado e registrar a referência medida

Registro publicado: [Issue #1](https://github.com/gretafg/awful-barenn/issues/1), fechado e associado à [tag `v0.1-baseline`](https://github.com/gretafg/awful-barenn/tree/v0.1-baseline).

## Escopo e cronologia

Registro retrospectivo da referência original. A execução e os registros locais precedem esta Issue; ela está sendo publicada agora, sem apresentar uma abertura anterior à implementação.

## Diagnóstico

O código original foi executado sem alterar `barenn.cpp`. A rede é 64 → 32 → 16 → 10, ReLU nas camadas ocultas, saída linear, SGD por amostra, 51 épocas, taxa 0.02 e semente 1337 reiniciada por camada. O perfil inicial Callgrind registrou 9503779205 instruções, com 85,81% atribuídas a `main` após inlining. A inspeção do código mostrou criação/cópia de vetores em forward/backward e cálculo de gradiente dos pixels que o chamador descarta.

## Hipótese

Reduzir temporários, cópias e o gradiente descartado, mantendo os cálculos e a sequência de SGD. Separar pesos e temporários também pode permitir inferência multicore com dados privados por imagem. A hipótese e sua avaliação foram registradas no bloco CPU; este marco preserva a referência.

## Métrica base e ambiente

Intel Core i5-1240P, WSL Ubuntu 22.04 e GCC 11. Compilação: `g++ -std=c++17 -O3 -g -ffp-contract=off barenn.cpp -o build/legacy`. Dados oficiais: 3823 imagens de treino e 1797 de teste.

A primeira execução individual informou treino de 0.5835 s, 1717/1797 acertos e perda média de teste impressa de 0.0432. Esse tempo não é mediana nem tempo total da aplicação: o cronômetro original inclui o treino e deixa leitura/avaliação de fora. A coleta posterior de cinco repetições por configuração está no registro CPU e não deve ser confundida com esta primeira observação.

## Reproduzir e consultar

- [Tag anotada v0.1-baseline](https://github.com/gretafg/awful-barenn/tree/v0.1-baseline): `make cpu` e `./build/legacy`.
- [Primeira saída original](https://github.com/gretafg/awful-barenn/blob/v0.1-baseline/results/local/legacy-first.txt).
- [Registro da referência](https://github.com/gretafg/awful-barenn/blob/main/docs/experiments/00-baseline.md).
- [Perfil Callgrind inicial, preservado no marco CPU](https://github.com/gretafg/awful-barenn/blob/v0.2-cpu/results/local/callgrind.legacy).
- [Resumo do perfil inicial](https://github.com/gretafg/awful-barenn/blob/v0.2-cpu/results/local/legacy-profile.txt).

Ir representa instruções, não segundos ou cache físico. As inferências de gargalo devem combinar o perfil com a inspeção e com os experimentos posteriores, sem atribuir todo o custo às alocações apenas pela concentração em `main`.
