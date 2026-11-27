# APS 2 — CPU e CUDA

Código original de [emil-freme/awful-barenn](https://github.com/emil-freme/awful-barenn) em `barenn.cpp`. A versão otimizada está em `src/`. Dados oficiais: 3823 imagens de treino e 1797 de teste; rede 64 → 32 → 16 → 10, 51 épocas, SGD por amostra, taxa 0.02 e semente 1337 por camada.

## Resultados

| Versão | Treino (s) | Inferência (s) | Coleta |
| --- | ---: | ---: | --- |
| Legado | 0.456937 | 0.126996 | CPU: mediana de 5 execuções, 100 passagens de inferência |
| CPU sequencial | 0.220282 | 0.135349 | Mesma coleta CPU |
| OpenMP, 4 threads na inferência | 0.222440 | 0.050758 | Mesma coleta CPU; treino sequencial |
| CUDA inicial | 1.234598 | 0.000066245 | Franky: mediana de 3 execuções, uma passagem de inferência |
| CUDA global | 0.974189 | 0.000062568 | Mesma coleta GPU |
| CUDA compartilhado, bloco 128 | 0.600314 | 0.000063185 | Mesma coleta GPU |

CPU local: i5-1240P, WSL Ubuntu 22.04, GCC 11. GPU em 27/11/2026: RTX A1000 de 8 GB, nó `compute20` do Franky, driver 570.124.06, nvcc 12.8.93 e GCC 14.2. Todas as medições usam 51 épocas. Na GPU, foram usados bloco 128, aquecimento separado e ordem alternada das versões. As nove execuções coincidiram com a referência CPU em perdas, escores, previsões e parâmetros: 1717/1797 acertos (95,55%).

O treino compartilhado foi **2,06× mais rápido que o CUDA inicial** e 1,62× que o global. A inferência global e compartilhada ficou próxima, sem ganho do compartilhado sobre o global. As coletas CPU local/GPU têm ambientes e números de passagens de inferência diferentes; seus tempos não demonstram speedup entre si.

## Registros

| Etapa | Discussão e evidências | Versão |
| --- | --- | --- |
| Original | [Issue #1](https://github.com/gretafg/awful-barenn/issues/1) | [v0.1-baseline](https://github.com/gretafg/awful-barenn/tree/v0.1-baseline) |
| CPU/OpenMP | [Issue #2](https://github.com/gretafg/awful-barenn/issues/2) | [v0.2-cpu](https://github.com/gretafg/awful-barenn/tree/v0.2-cpu) |
| CUDA funcional | [Issue #3](https://github.com/gretafg/awful-barenn/issues/3) | [v0.3-cuda-funcional](https://github.com/gretafg/awful-barenn/tree/v0.3-cuda-funcional) |
| Comparação CUDA e profiling | [Issue #3](https://github.com/gretafg/awful-barenn/issues/3) | [v0.4-cuda-medido](https://github.com/gretafg/awful-barenn/tree/v0.4-cuda-medido) |

## Análise da GPU

O Nsight Systems foi coletado separadamente, com uma época por versão. O treino inicial lançou 3823 kernels; global e compartilhado lançaram um cada. Seus kernels de treino somaram, respectivamente, 30,433 ms, 25,387 ms e 15,709 ms nesse recorte instrumentado. Isso sustenta reduzir lançamentos e reutilizar pesos compartilhados. O treino usa um bloco para preservar as dependências do SGD, limitando o uso de vários SMs.

Na coleta sem profiler, a preparação mediana do compartilhado foi 1,643844 s, incluindo H2D de 0,222816 ms; D2H foi 0,030560 ms. A preparação supera o treino e não pode ser atribuída toda à transferência dos dados. O Nsight Compute foi tentado, mas retornou `ERR_NVGPUCTRPERM`: o usuário não tem acesso aos contadores GPU. Não foram obtidas métricas de ocupação real, cache ou conflitos de bancos. Tempos individuais, dispersão, perfis e limitações estão na Issue #3.

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
nvcc -std=c++17 -O3 -lineinfo --fmad=false -arch=sm_75 -Xcompiler=-fopenmp -Xcompiler=-ffp-contract=off src/cuda_main.cu -o build/barenn_cuda
srun --partition=merry_gpu --gres=gpu:1 --ntasks=1 --cpus-per-task=1 --mem=1G --time=00:02:00 build/barenn_cuda --backend cuda-shared --block 128
```

O padrão é 51 épocas e uma passagem de inferência. Outros backends: `cuda-naive` e `cuda-global`. Use `--output resultado.json` para salvar perdas, escores, previsões e parâmetros.

As Issues são registros retrospectivos. Os marcos iniciais foram reorganizados para manter apenas código, dados e instruções; isso não altera as datas dos experimentos nem comprova Issue aberta antes da implementação. O marco `v0.4-cuda-medido` registra a coleta de 27/11/2026, com o mesmo código de cálculo do commit `6cdd735`.
