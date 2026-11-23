# APS 2 — CPU otimizada

O código original de [emil-freme/awful-barenn](https://github.com/emil-freme/awful-barenn) está preservado em `barenn.cpp`. A versão otimizada usa parâmetros contíguos, temporários reutilizados e inferência OpenMP; o treino SGD continua sequencial.

```bash
mkdir -p build
g++ -std=c++17 -O3 -ffp-contract=off -fopenmp src/main.cpp -o build/barenn_cpu
./build/barenn_cpu --backend legacy
./build/barenn_cpu --backend seq
./build/barenn_cpu --backend omp --threads 4
```

| Versão | Treino mediano (s) | Inferência mediana (s, 100 passagens) |
| --- | ---: | ---: |
| Legado | 0.456937 | 0.126996 |
| CPU sequencial | 0.220282 | 0.135349 |
| Inferência OpenMP, 4 threads | 0.222440 | 0.050758 |

Cinco repetições por configuração, em Intel Core i5-1240P, WSL Ubuntu 22.04 e GCC 11. Treino completo de 51 épocas; perdas, escores e previsões coincidiram, com 1717/1797 acertos. A inferência sequencial não melhorou nessa coleta.

Registros: [referência](https://github.com/gretafg/awful-barenn/issues/1) e [CPU/OpenMP](https://github.com/gretafg/awful-barenn/issues/2). Os registros foram publicados após a implementação.
