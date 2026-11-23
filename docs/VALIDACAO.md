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
