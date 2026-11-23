#!/usr/bin/env bash
# Uma execucao do backend CUDA otimizado; nao submete a bateria completa.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
command -v srun >/dev/null || { echo "Execute no terminal do Franky."; exit 1; }
if ! command -v nvcc >/dev/null; then
    module load "${CUDA_MODULE:-cuda/12.8.1}"
fi
# Os headers de desenvolvimento estao no no de acesso.
make cuda
OUT="${1:-results/cluster/minimo-$(date +%Y%m%d-%H%M%S)}"
if [[ -e "$OUT" ]]; then
    echo "Saida ja existe: $OUT. Use uma pasta nova para preservar a evidencia."
    exit 1
fi
mkdir -p "$OUT"
srun --job-name=aps2-minimo --partition="${GPU_PARTITION:-merry_gpu}" \
    --gres=gpu:1 --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=1G \
    --time="${GPU_TIME:-00:02:00}" \
    bash -c 'hostname; nvidia-smi; exec "$@"' _ \
    build/barenn_cuda --backend cuda-shared --block 128 --epochs 51 \
    --infer-repeats 1 --output "$OUT/cuda-shared.json" 2>&1 | tee "$OUT/execucao.log"
echo "Resultado salvo em $OUT. Uma execucao nao estima speedup nem dispersao."
