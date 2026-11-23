# APS 2 — Referência

Código original de [emil-freme/awful-barenn](https://github.com/emil-freme/awful-barenn), preservado em `barenn.cpp`.

```bash
mkdir -p build
g++ -std=c++17 -O3 -ffp-contract=off barenn.cpp -o build/legacy
./build/legacy
```

Dados: `optdigits.tra` e `optdigits.tes`. Rede 64 → 32 → 16 → 10, 51 épocas, SGD por amostra, taxa 0.02 e semente 1337 por camada.

Primeira execução local: 1717/1797 acertos e treino de 0.5835 s. É uma observação individual, não a mediana das medições posteriores.

[Issue #1: referência e medições](https://github.com/gretafg/awful-barenn/issues/1).
