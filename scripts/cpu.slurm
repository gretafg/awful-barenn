#!/bin/bash
#SBATCH --job-name=aps2-cpu
#SBATCH --partition=pluton_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submeta na raiz do repositorio}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
OUT="results/cluster/cpu-${SLURM_JOB_ID}"
mkdir -p "$OUT"
# Compile no no de acesso antes de submeter; herde seu ambiente de runtime.
for binary in build/legacy build/barenn_cpu build/validate; do
    test -x "$binary" || { echo "Falta $binary. Compile no no de acesso com make cpu build/validate."; exit 1; }
done
{
    hostname
    ./build/validate
    python3 -m unittest discover -s tests -p 'test_*.py'
} 2>&1 | tee "$OUT/build-and-tests.log"
python3 scripts/run_suite.py --out "$OUT/full" --repetitions 5 --infer-repeats 100
bash scripts/profile_cpu.sh "$OUT/profiles"
python3 scripts/report.py "$OUT/full" --out "$OUT/report"
