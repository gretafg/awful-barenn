#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
command -v sbatch >/dev/null || { echo "Execute este comando no terminal do Franky."; exit 1; }
# Os nos de computacao observados nao possuem todos os headers de C/C++.
# A compilacao fica no no de acesso; o sbatch herda CUDA e bibliotecas.
if ! command -v nvcc >/dev/null; then
    module load "${CUDA_MODULE:-cuda/12.8.1}"
fi
make cpu cuda build/validate
mkdir -p logs results/cluster
cpu_job="$(sbatch --parsable --partition="${CPU_PARTITION:-pluton_cpu}" --time="${CPU_TIME:-00:30:00}" scripts/cpu.slurm)"
gpu_job="$(sbatch --parsable --partition="${GPU_PARTITION:-pluton_gpu}" --time="${GPU_TIME:-00:50:00}" scripts/gpu.slurm)"
printf 'CPU: %s\nGPU: %s\nAcompanhar: squeue -u %s\nLogs: %s/logs\n' "$cpu_job" "$gpu_job" "$USER" "$ROOT"
printf '%s\n' "Depois: python3 scripts/report.py results/cluster/cpu-${cpu_job%%;*} results/cluster/gpu-${gpu_job%%;*} --out results/cluster/relatorio"
