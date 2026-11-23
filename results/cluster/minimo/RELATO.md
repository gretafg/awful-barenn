# Execução mínima no Franky — 23/11/2026

Foi executado somente o backend otimizado `cuda-shared`, uma vez, sem profilers ou repetições de desempenho, conforme pedido da usuária. Não houve commits, tags, pushes ou publicação de Issues.

- Usuário: `gretadfgg`; host de acesso: `10.103.1.38`.
- Pasta remota: `/home/gretadfgg/aps2-20261123-1426`.
- Nó observado na inspeção imediatamente anterior: `compute20`, NVIDIA RTX A1000, 8 GB, driver 570.124.06. A execução usou a mesma partição; o nome do nó da execução não foi gravado pelo programa.
- Compilador: NVIDIA nvcc 12.8.93; módulo `cuda/12.8.1`; flags registradas em `../logs/build-minimo.log`.
- Rede original 64 → 32 → 16 → 10; treino com 3823 amostras, 51 épocas; teste com 1797 imagens e uma passagem; bloco de 128 threads.

| Medida | Resultado |
| --- | ---: |
| Treino, tempo observado pelo host | 0,768943237 s |
| Treino, eventos CUDA | 768,950012 ms |
| Inferência, tempo observado pelo host | 0,000064663 s |
| Inferência, eventos CUDA | 0,060416 ms |
| Preparação, incluindo contexto, alocações e H2D | 1,711880743 s |
| H2D, incluído na preparação | 2,432448 ms |
| D2H | 0,034912 ms |
| Acertos | 1717/1797 (95,55%) |
| Perda de teste | 0,04319660732456998 |

Os tempos das regiões não representam o tempo total do processo: não incluem, por exemplo, leitura dos arquivos e exportação do JSON. Uma execução não estima dispersão nem comprova ganho de velocidade. Não comparar diretamente com os tempos locais de CPU, obtidos em outra máquina e com 100 passagens na inferência.

## Correção sem novas execuções no cluster

O arquivo `cuda-shared.json` foi baixado e comparado com as referências já existentes:

```text
python scripts/compare.py results/local/reference.json results/cluster/minimo/cuda-shared.json
PASS: epoch_losses_max_abs=0.0; scores_max_abs=0.0

python scripts/compare.py results/local/cpu.json results/cluster/minimo/cuda-shared.json
PASS: epoch_losses_max_abs=0.0; scores_max_abs=0.0; parameters_max_abs=0.0
```

O comparador também verificou previsões e matriz de confusão. Não foram executados Compute Sanitizer, outros backends, varredura de blocos ou perfis. Isso é uma execução funcional completa de uma configuração, não a coleta experimental completa da APS.

## Comandos que funcionaram

No nó de acesso, dentro da pasta da APS:

```bash
module load cuda/12.8.1
make cuda
mkdir -p results/cluster/minimo
srun --job-name=aps2-minimo --partition=merry_gpu --gres=gpu:1 \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=1G --time=00:02:00 \
  build/barenn_cuda --backend cuda-shared --block 128 --epochs 51 \
  --infer-repeats 1 --output results/cluster/minimo/cuda-shared.json
```

Binário executado, SHA-256: `18b651c998b289ecab2bfc6c6fb169bdd1e9cb9dd30075890a4f2da3239d24bd`.

Os jobs iniciais 30679 (CPU) e 30680 (GPU) falharam durante a preparação, antes das medições. Seus logs foram preservados em `../logs/`. A compilação no nó de acesso resolveu o necessário para esta execução mínima. A última consulta à fila não mostrou nenhum job da usuária.
