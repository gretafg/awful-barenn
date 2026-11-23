# Reproduzir a APS 2

Esta cópia reúne código e resultados já obtidos. A implementação foi organizada em commits locais, sem push. A parte de publicação e Issues continua pendente: veja [governança](docs/GOVERNANCA.md).

## Validar CPU

Na raiz do projeto, em Linux ou WSL:

```bash
make cpu test
./build/barenn_cpu --backend legacy --output referencia.json
./build/barenn_cpu --backend seq --output cpu.json
./build/barenn_cpu --backend omp --threads 4 --output omp.json
python3 scripts/compare.py referencia.json cpu.json
python3 scripts/compare.py cpu.json omp.json
```

`make test-kernel-logic` exercita a lógica dos kernels com threads CPU; não simula memória real ou races na GPU. `make cuda` compila o executável CUDA e não executa kernels.

## Executar somente o CUDA otimizado no Franky

Transfira a pasta local ou gere um pacote com `python scripts/package.py ../aps2-entrega-local.zip`. Clonar apenas o fork remoto ainda não traz as alterações locais.

No nó de acesso, dentro da pasta transferida:

```bash
module load cuda/12.8.1
bash scripts/run_franky_minimo.sh
```

O script compila no nó de acesso e pede uma GPU ao SLURM. Executa apenas `cuda-shared`, uma vez, com 51 épocas, bloco 128 e uma passagem de inferência. A saída fica em `results/cluster/minimo-DATA-HORA/`. Também aceita um diretório novo como primeiro argumento.

É o mesmo backend e configuração da execução já preservada em [results/cluster/minimo/](results/cluster/minimo/RELATO.md). O novo lançador registra também hostname e nvidia-smi; ele foi revisado e sua sintaxe foi conferida localmente, sem nova execução no Franky.

## Coleta completa opcional

O comando abaixo submete **duas baterias**, CPU e GPU, incluindo repetições e perfis. Não é necessário para repetir apenas o otimizado:

```bash
bash scripts/submit_franky.sh
```

Antes de submeter, ele carrega CUDA e compila os executáveis no nó de acesso. Os jobs herdam o ambiente e executam os binários nos nós de computação. Essa separação corrige a tentativa inicial, que falhou porque faltavam headers nos nós e havia diferença no ambiente de módulos. A bateria ajustada ainda não foi executada no Franky.

Padrões: `pluton_cpu` por 30 minutos e `pluton_gpu` por 50 minutos. É possível indicar `CPU_PARTITION`, `GPU_PARTITION`, `CPU_TIME`, `GPU_TIME` e `CUDA_MODULE` no ambiente. Confira `sinfo` e `module avail cuda` no cluster.

Os logs ficam em `logs/` e as medições em `results/cluster/cpu-JOB` e `results/cluster/gpu-JOB`. Acompanhe com `squeue -u "$USER"`. Cada conjunto completo deve ter `full/SUCCESS`; isso não atesta que todos os profilers estavam disponíveis. Consulte os arquivos de pendência e os logs.

Para consolidar uma coleta completa, substitua os números pelos jobs reais:

```bash
python3 scripts/report.py results/cluster/cpu-JOB results/cluster/gpu-JOB --out results/cluster/relatorio
```

A execução mínima preservada não possui `SUCCESS` de uma bateria completa e não deve ser apresentada como tal. Seus dados estão no JSON e no registro específico. Não aumente tolerâncias para ocultar falhas.

Os scripts não criam commits, tags, Issues, PRs ou envios ao Blackboard.
