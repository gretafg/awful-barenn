#!/bin/bash
#SBATCH --job-name=aps2-gpu
#SBATCH --partition=pluton_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=00:50:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submeta na raiz do repositorio}"
OUT="results/cluster/gpu-${SLURM_JOB_ID}"
mkdir -p "$OUT"
# O submit carrega CUDA e compila no no de acesso, antes do sbatch.
command -v nvcc >/dev/null || { echo "Carregue CUDA no no de acesso antes de submeter."; exit 1; }
test -x build/barenn_cuda || { echo "Execute make cuda no no de acesso antes de submeter."; exit 1; }
hostname | tee "$OUT/hostname.txt"
nvcc --version | tee "$OUT/nvcc.txt"
nvidia-smi | tee "$OUT/gpu.txt"
python3 scripts/run_suite.py --gpu --quick --out "$OUT/quick" --repetitions 1 --infer-repeats 1
if command -v compute-sanitizer >/dev/null; then
    for backend in cuda-naive cuda-global cuda-shared; do
        compute-sanitizer --tool memcheck --error-exitcode 97 build/barenn_cuda \
            --backend "$backend" --train "$OUT/quick/train-small.csv" \
            --test "$OUT/quick/test-small.csv" --epochs 2 \
            > "$OUT/memcheck-$backend.log" 2>&1
    done
    for tool in racecheck synccheck; do
        compute-sanitizer --tool "$tool" --error-exitcode 98 build/barenn_cuda \
            --backend cuda-shared --train "$OUT/quick/train-small.csv" \
            --test "$OUT/quick/test-small.csv" --epochs 2 \
            > "$OUT/$tool.log" 2>&1
    done
else
    echo "PENDENTE: Compute Sanitizer nao encontrado." | tee "$OUT/SANITIZER_PENDING"
fi
python3 scripts/run_suite.py --gpu --out "$OUT/full" --repetitions 5 --infer-repeats 100
bash scripts/profile_gpu.sh "$OUT/profiles"
python3 scripts/report.py "$OUT/full" --out "$OUT/report"
