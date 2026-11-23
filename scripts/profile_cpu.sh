#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT="${1:?Informe o diretorio de saida}"
mkdir -p "$OUT"
if ! command -v valgrind >/dev/null; then
    echo "PENDENTE: valgrind nao encontrado." | tee "$OUT/PROFILE_PENDING"
    exit 0
fi
for backend in legacy seq; do
    valgrind --tool=callgrind --cache-sim=yes --branch-sim=yes \
        --callgrind-out-file="$OUT/callgrind-$backend" \
        build/barenn_cpu --backend "$backend" --epochs 3 \
        > "$OUT/$backend.stdout" 2> "$OUT/$backend.stderr"
    callgrind_annotate --threshold=99 --auto=no "$OUT/callgrind-$backend" \
        > "$OUT/summary-$backend.txt"
done
echo "Perfis usam 3 epocas para conter o custo de instrumentacao." > "$OUT/scope.txt"
