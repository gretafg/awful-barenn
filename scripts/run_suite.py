#!/usr/bin/env python3
"""Run measured experiments, reject numerical regressions, preserve raw outputs."""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from compare import compare

ROOT = Path(__file__).resolve().parents[1]

def command(args):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    return {"command": args, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--infer-repeats", type=int, default=100)
    args = p.parse_args()
    if args.repetitions < 1 or args.infer_repeats < 1:
        p.error("Repeticoes devem ser positivas")
    os.chdir(ROOT)
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/"manifest.json").exists():
        p.error("Diretorio ja contem uma execucao; escolha outro --out")
    train, test = ROOT/"optdigits.tra", ROOT/"optdigits.tes"
    if args.quick:
        train, test = output/"train-small.csv", output/"test-small.csv"
        train.write_text("\n".join((ROOT/"optdigits.tra").read_text().splitlines()[:33])+"\n")
        test.write_text("\n".join((ROOT/"optdigits.tes").read_text().splitlines()[:17])+"\n")
    epochs = 2 if args.quick else 51
    binary = ROOT/"build"/("barenn_cuda" if args.gpu else "barenn_cpu")
    common = ["--train", str(train), "--test", str(test), "--epochs", str(epochs),
              "--infer-repeats", str(args.infer_repeats)]
    manifest = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(), "python": sys.version,
        "scope": "quick" if args.quick else "full",
        "git": command(["git","rev-parse","HEAD"]),
        "git_status": command(["git","status","--short"]),
        "compiler": command(["g++","--version"]),
        "cpu": command(["lscpu"]),
        "slurm": {k:v for k,v in os.environ.items() if k.startswith("SLURM_")},
        "openmp": {k:v for k,v in os.environ.items() if k.startswith("OMP_")},
        "dataset_sha256": {"train": sha(train), "test": sha(test)},
        "binary_sha256": sha(binary),
        "source_sha256": {str(f.relative_to(ROOT)):sha(f) for f in
                          [ROOT/"Makefile",ROOT/"barenn.cpp",*sorted((ROOT/"src").glob("*"))]},
        "repetitions": args.repetitions, "epochs": epochs,
        "infer_repeats": args.infer_repeats,
        "tolerance": {"atol": 1e-6, "rtol": 1e-5},
    }
    if args.gpu:
        manifest["cuda_compiler"] = command(["nvcc","--version"])
        manifest["gpu"] = command(["nvidia-smi"])
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    (output/"source.diff").write_text(command(["git","diff","HEAD"])["stdout"])
    ref_path = output/"reference.json"
    ref_cmd = [str(binary),"--backend","legacy",*common,"--output",str(ref_path)]
    result = command(ref_cmd)
    (output/"reference.log").write_text(json.dumps(result,indent=2))
    if result["returncode"]:
        raise RuntimeError("Referencia falhou: "+result["stderr"])
    reference = json.loads(ref_path.read_text())
    parameter_path = output/"cpu-parameters.json"
    result = command([str(binary),"--backend","seq",*common,"--output",str(parameter_path)])
    (output/"cpu-parameters.log").write_text(json.dumps(result,indent=2))
    if result["returncode"]:
        raise RuntimeError("Referencia CPU falhou: "+result["stderr"])
    parameter_reference = json.loads(parameter_path.read_text())
    compare(reference,parameter_reference)
    if args.gpu:
        configs = [(backend,block) for backend in ("cuda-naive","cuda-global","cuda-shared")
                   for block in (64,128,256)]
    else:
        configs = [("legacy",1),("seq",1),*[("omp",n) for n in (1,2,4,8)]]
    work = [(backend,n,r) for backend,n in configs for r in range(args.repetitions)]
    random.Random(1337).shuffle(work)
    # Warmup igual para cada configuracao, fora das amostras registradas.
    for backend,n in configs:
        option = "--block" if backend.startswith("cuda") else "--threads"
        result = command([str(binary),"--backend",backend,option,str(n),*common])
        if result["returncode"]:
            raise RuntimeError("Warmup falhou: "+result["stderr"])
    rows = []
    for backend,n,r in work:
        option = "--block" if backend.startswith("cuda") else "--threads"
        name = f"{backend}-{n}-r{r}"
        path = output/(name+".json")
        result = command([str(binary),"--backend",backend,option,str(n),*common,"--output",str(path)])
        (output/(name+".log")).write_text(json.dumps(result,indent=2))
        if result["returncode"]:
            raise RuntimeError(name+" falhou: "+result["stderr"])
        data = json.loads(path.read_text())
        validation = compare(reference,data)
        if backend != "legacy":
            validation.update(compare(parameter_reference,data))
        (output/(name+".validation.json")).write_text(json.dumps(validation,indent=2))
        row = {k:data[k] for k in ("backend","threads","block","train_s","infer_s","setup_s",
                                  "h2d_ms","d2h_ms","train_device_ms","infer_device_ms","correct","test_loss")}
        row.update(repetition=r,artifact=path.name)
        rows.append(row)
        print(f"PASS {name}: treino={data['train_s']:.6f}s inferencia={data['infer_s']:.6f}s",flush=True)
        with (output/"measurements.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output/"SUCCESS").write_text("Todas as execucoes foram comparadas com a referencia.\n")
    print(f"Resultados: {output}")

if __name__ == "__main__":
    main()
