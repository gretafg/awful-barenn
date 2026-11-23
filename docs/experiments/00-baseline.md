# Referência do legado

Registro iniciado localmente e publicado posteriormente na [Issue #1](https://github.com/gretafg/awful-barenn/issues/1), associado à tag `v0.1-baseline`. A Issue é retrospectiva e não comprova abertura anterior às alterações no GitHub.

## Configuração

Código de origem: commit `27f88c4` do repositório emil-freme/awful-barenn. Dados oficiais: 3823 imagens de treino e 1797 de teste. Treino com 51 épocas, SGD por amostra na ordem do arquivo, taxa 0.02 e semente 1337 por camada.

Primeira execução em WSL Ubuntu 22.04 sobre Intel Core i5-1240P. Compilação: `g++ -std=c++17 -O3 -g -ffp-contract=off barenn.cpp -o build/legacy`.

## Observação inicial

A primeira execução completa terminou com 1717 acertos em 1797 imagens, perda média de teste impressa de 0.0432 e tempo de treino de 0.5835 s. Esse tempo é uma observação individual, não uma estimativa estatística de ganho. Log: `results/local/legacy-first.txt`.

O cronômetro original inclui apenas o treino. A leitura dos dados e a avaliação estão fora desse intervalo. As comparações posteriores devem manter essa distinção.

## Investigação

Coletar Callgrind do executável original. A hipótese inicial, baseada na leitura do código, é que buffers temporários, cópias e o gradiente da primeira entrada descartado consomem trabalho evitável. Confirmar a relevância com o perfil antes de interpretar o ganho.

## Correção de referência

Preservar o arquivo `barenn.cpp` integralmente. As versões instrumentadas vão reutilizar suas classes por inclusão, para comparar saídas e treino sem reescrever a referência matemática.
