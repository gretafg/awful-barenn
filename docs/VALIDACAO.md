# Verificações disponíveis

## Executadas localmente

- Compilação CPU com GCC 11, C++17, O3, OpenMP e contração de ponto flutuante desabilitada.
- Comparação de 512 passos de SGD, forward e perdas contra as classes originais incluídas diretamente de `barenn.cpp`.
- Uma verificação independente de gradiente por diferenças finitas.
- Inferência com 1, 2, 4 e 8 threads, incluindo lote de 33 imagens.
- Treino completo de 51 épocas e comparação de todos os escores, perdas, previsões e matriz de confusão.
- Rejeição de argumentos e arquivos de entrada inválidos.
- Rejeição de NaN, dimensões truncadas e previsões divergentes pelo comparador.
- Trinta execuções medidas, com cinco repetições por configuração e comparação com a referência.
- Execução dos testes de CPU com AddressSanitizer e UndefinedBehaviorSanitizer. Log em `results/local/asan.log`.
- Compilação dos kernels com NVIDIA nvcc 12.8.93 para sm_75 e PTX, sem necessidade de uma GPU local. Log em `results/local/cuda-build.log`.
- Execução da mesma lógica dos kernels em CPU, com threads e barreiras, comparando parâmetros e escores nos tamanhos 64, 128 e 256. A inferência usa 17 imagens para exercitar o último bloco incompleto.
- Teste do fluxo de coleta em modo pequeno, incluindo exportação e comparação de parâmetros.

## Comandos de reprodução em Linux

```bash
make cpu test
make test-kernel-logic
make cuda
g++ -std=c++17 -O1 -g -ffp-contract=off -fopenmp -fsanitize=address,undefined -fno-omit-frame-pointer tests/validate.cpp -o build/validate_asan
./build/validate_asan
```

## A executar no Franky

Em 23/11/2026, `cuda-shared` executou 51 épocas e uma passagem de inferência na RTX A1000 do Franky, com bloco 128. Perdas, escores, previsões, matriz de confusão e parâmetros coincidiram exatamente com as referências locais, sem executar novamente o legado no cluster. Saída e comparação em `results/cluster/minimo/`.

Continuam pendentes execução dos outros backends na GPU física, Compute Sanitizer, Nsight Systems/Compute e medições repetidas para escolher o tamanho de bloco. A usuária pediu somente uma execução otimizada, sem a bateria de experimentos. O teste de lógica em CPU não modela execução SIMT, coerência real, conflitos de bancos ou races na GPU.

Os scripts SLURM realizam os testes pequenos antes da coleta completa. Se um teste ou a comparação falhar, o job encerra com erro e preserva os logs. Falhas ou ausência de profilers geram registros explícitos de pendência; não são transformadas em resultados.

## Verificação do novo clone para entrega

Após clonar o fork em `awful-barenn`, os testes foram repetidos localmente, sem nova execução no Franky:

- `make cpu test` e, após ajustar os lançadores, `make test`: passaram a validação C++ de 512 passos SGD/gradiente/inferência e os quatro testes Python, incluindo treino completo e sintaxe dos scripts.
- `make cuda` com nvcc 12.8.93: compilação concluída, sem execução dos kernels na GPU local.
- `make test-kernel-logic`: lógica global/shared/naive e último grupo incompleto passaram em CPU para blocos 64/128/256.
- O executável compilado pelo nvcc foi executado com `--backend seq`, na CPU, e comparado à referência: perdas, escores e parâmetros coincidiram exatamente. Esse teste não é uma nova execução CUDA.
- O JSON da execução real anterior do Franky foi novamente comparado com as referências de legado e CPU: diferença máxima zero em perdas, escores e parâmetros disponíveis; previsões e matriz também coincidiram.
- As fontes de cálculo migradas coincidem com a preparação anterior após normalização dos finais de linha Windows/Linux. O legado não tem alterações em relação ao fork. Os hashes foram registrados para distinguir diferenças de arquivo das contas preservadas.
- Os links locais da documentação e o pacote de transferência foram conferidos. A evidência GPU e os logs da tentativa inicial agora são versionados; binários, chaves, `.git` e saídas novas de coleta não entram no ZIP.

Logs e hashes: [results/local/entrega/](../results/local/entrega/manifest.json). A validação original de GPU está em [validation.json](../results/cluster/minimo/validation.json). As medições históricas não foram substituídas pelos tempos dos testes atuais.

Os lançadores passaram na verificação de sintaxe. A compilação foi movida para o nó de acesso, com execução nos nós pelo SLURM e ambiente herdado. Essa correção trata a falha observada na primeira tentativa do cluster; a bateria completa ajustada ainda não foi executada.
