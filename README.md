# APS 2 — CPU e CUDA

Código original de [emil-freme/awful-barenn](https://github.com/emil-freme/awful-barenn) em `barenn.cpp`. A versão otimizada está em `src/`. Dados oficiais: 3823 imagens de treino e 1797 de teste; rede 64 → 32 → 16 → 10, 51 épocas, SGD por amostra, taxa 0.02 e semente 1337 por camada.

## Resultados

| Versão | Treino (s) | Inferência (s) | Coleta |
| --- | ---: | ---: | --- |
| Legado | 0.456937 | 0.126996 | CPU: mediana de 5 execuções, 100 passagens de inferência |
| CPU sequencial | 0.220282 | 0.135349 | Mesma coleta CPU |
| OpenMP, 4 threads na inferência | 0.222440 | 0.050758 | Mesma coleta CPU; treino sequencial |
| CUDA inicial | — | — | Implementado; sem medição em GPU |
| CUDA global | — | — | Implementado; sem medição em GPU |
| CUDA compartilhado, bloco 128 | 0.768943 | 0.000064663 | Franky: uma execução, uma passagem de inferência |

CPU: i5-1240P, WSL Ubuntu 22.04, GCC 11. GPU: RTX A1000 observada na inspeção anterior, nvcc 12.8.93. Foram 1717/1797 acertos (95,55%). A execução CUDA coincidiu com as referências em perdas, escores e parâmetros. Preparação GPU: 1.711881 s, incluindo H2D; D2H: 0.034912 ms. As coletas CPU/GPU têm ambientes e escopos diferentes e não demonstram speedup entre si. Faltam comparações e perfis GPU.

## Executar CPU

```bash
mkdir -p build
g++ -std=c++17 -O3 -ffp-contract=off -fopenmp src/main.cpp -o build/barenn_cpu
./build/barenn_cpu --backend omp --threads 4
```

Também estão disponíveis `--backend legacy` e `--backend seq`.

## Executar CUDA no Franky

Compile no nó de acesso e execute pelo SLURM:

```bash
module load cuda/12.8.1
mkdir -p build
nvcc -std=c++17 -O3 --fmad=false -arch=sm_75 -Xcompiler=-fopenmp -Xcompiler=-ffp-contract=off src/cuda_main.cu -o build/barenn_cuda
srun --partition=merry_gpu --gres=gpu:1 --ntasks=1 --cpus-per-task=1 --mem=1G --time=00:02:00 build/barenn_cuda --backend cuda-shared --block 128
```

O padrão é 51 épocas e uma passagem de inferência. Outros backends: `cuda-naive` e `cuda-global`. Use `--output resultado.json` para salvar perdas, escores, previsões e parâmetros.

## Registros

| Etapa | Discussão e evidências | Versão |
| --- | --- | --- |
| Original | [Issue #1](https://github.com/gretafg/awful-barenn/issues/1) | [v0.1-baseline](https://github.com/gretafg/awful-barenn/tree/v0.1-baseline) |
| CPU/OpenMP | [Issue #2](https://github.com/gretafg/awful-barenn/issues/2) | [v0.2-cpu](https://github.com/gretafg/awful-barenn/tree/v0.2-cpu) |
| CUDA funcional | [Issue #3](https://github.com/gretafg/awful-barenn/issues/3) | [v0.3-cuda-funcional](https://github.com/gretafg/awful-barenn/tree/v0.3-cuda-funcional) |

As Issues são registros retrospectivos. As três versões foram reorganizadas para manter apenas código, dados e instruções; isso não altera as datas dos experimentos nem comprova Issue aberta antes da implementação.
