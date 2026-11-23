#!/usr/bin/env python3
"""Summarize real validated runs; figures are optional when matplotlib is absent."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("directories",type=Path,nargs="+")
    p.add_argument("--out",type=Path,required=True)
    args = p.parse_args()
    out = args.out
    out.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for directory in args.directories:
        # Aceita tanto a raiz do job quanto o diretorio full.
        if not (directory/"manifest.json").exists() and (directory/"full").is_dir():
            directory=directory/"full"
        if not (directory/"SUCCESS").is_file():
            raise SystemExit(f"Execucao ausente, incompleta ou nao validada: {directory}")
        manifest=json.loads((directory/"manifest.json").read_text())
        rows=list(csv.DictReader((directory/"measurements.csv").open()))
        groups=defaultdict(list)
        for row in rows:
            n=row["block"] if row["backend"].startswith("cuda") else row["threads"]
            groups[(row["backend"],n)].append(row)
        for (backend,n),group in groups.items():
            record={"environment":str(directory),"backend":backend,"configuration":n,"n":len(group),
                    "scope":manifest["scope"],"epochs":manifest["epochs"],
                    "infer_repeats":manifest["infer_repeats"]}
            for key in ("train_s","infer_s","setup_s","h2d_ms","d2h_ms","train_device_ms","infer_device_ms"):
                values=[float(row[key]) for row in group]
                record[key]=statistics.median(values)
                record[key+"_min"]=min(values);record[key+"_max"]=max(values)
            # Soma das medianas e estimativa por componentes, nao tempo total observado.
            record["gpu_components_s"]=record["setup_s"]+record["train_s"]+record["infer_s"]+record["d2h_ms"]/1000
            summaries.append(record)
    with (out/"summary.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]))
        writer.writeheader();writer.writerows(summaries)
    text=["# Resultados medidos","",
          "Medianas de execucoes que passaram pela comparacao numerica. Intervalo entre colchetes: minimo e maximo.",
          "Inferencia reporta o tempo de todas as repeticoes indicadas, sem dividir silenciosamente pelo lote.",
          "Comparacoes entre ambientes ou nos diferentes nao isolam o efeito da implementacao.","",
          "| Ambiente | Versao | Configuracao | n | Treino em s | Inferencia em s |",
          "| --- | --- | --- | ---: | --- | --- |"]
    for s in summaries:
        def fmt(key):
            return f"{s[key]:.6f} [{s[key+'_min']:.6f}, {s[key+'_max']:.6f}]"
        text.append(f"| {s['environment']} | {s['backend']} | {s['configuration']} | {s['n']} | {fmt('train_s')} | {fmt('infer_s')} |")
    text.extend(["","## Escopo e interpretacao","",
                 "Treino e inferencia sao intervalos separados. Na GPU, setup inclui contexto, alocacoes e envio inicial. "
                 "O CSV tambem mostra tempos de device e retorno. gpu_components_s soma medianas de componentes e nao substitui "
                 "uma medicao de tempo total da aplicacao.",
                 "","Os perfis devem ser interpretados com os logs de hardware, comandos e flags do mesmo experimento. "
                 "Nenhuma medicao em CPU local comprova ganho no Franky ou em CUDA."])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
        environments={name:i+1 for i,name in enumerate(dict.fromkeys(s["environment"] for s in summaries))}
        labels=[f"{s['backend']} ({s['configuration']})\nambiente {environments[s['environment']]}" for s in summaries]
        for ax,key,title in zip(axes,("train_s","infer_s"),("Treino","Inferencia")):
            values=[s[key] for s in summaries]
            errors=[[s[key]-s[key+"_min"] for s in summaries],
                    [s[key+"_max"]-s[key] for s in summaries]]
            ax.bar(range(len(values)),values,yerr=errors,capsize=3,color="#176B87")
            ax.set_xticks(range(len(values)),labels,rotation=65,ha="right",fontsize=7)
            ax.set_ylabel("segundos");ax.set_title(title+" — mediana e intervalo min/max")
            ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
        fig.savefig(out/"times.png",dpi=180)
        plt.close(fig)
        text.extend(["","![Tempos medidos](times.png)"])
    except ImportError:
        text.extend(["","Grafico pendente: instale matplotlib para gerar a figura a partir do CSV."])
    (out/"report.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    print(out/"report.md")

if __name__=="__main__":
    main()
