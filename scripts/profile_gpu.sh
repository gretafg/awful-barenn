#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT="${1:?Informe o diretorio de saida}"
mkdir -p "$OUT"
if command -v nsys >/dev/null; then
    for backend in cuda-naive cuda-global cuda-shared; do
        if nsys profile --force-overwrite=true --trace=cuda,osrt --stats=true \
            -o "$OUT/$backend" build/barenn_cuda --backend "$backend" --epochs 1 \
            > "$OUT/nsys-$backend.log" 2>&1; then
            echo "OK $backend" >> "$OUT/nsys-status.txt"
        else
            echo "PENDENTE $backend: consulte o log." >> "$OUT/nsys-status.txt"
        fi
    done
else
    echo "PENDENTE: Nsight Systems nao encontrado." > "$OUT/NSYS_PENDING"
fi
if command -v ncu >/dev/null; then
    for backend in cuda-global cuda-shared; do
        if ncu --set full --launch-count 1 --target-processes all \
            --force-overwrite --export "$OUT/ncu-$backend" \
            build/barenn_cuda --backend "$backend" --epochs 1 \
            > "$OUT/ncu-$backend.log" 2>&1; then
            echo "OK $backend" >> "$OUT/ncu-status.txt"
        else
            echo "PENDENTE $backend: permissoes de contadores ou outra falha; consulte o log." >> "$OUT/ncu-status.txt"
        fi
    done
else
    echo "PENDENTE: Nsight Compute nao encontrado." > "$OUT/NCU_PENDING"
fi
echo "Perfis usam uma epoca; tempos do profiler nao entram nas medianas." > "$OUT/scope.txt"
