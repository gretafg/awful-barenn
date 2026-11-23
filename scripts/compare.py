#!/usr/bin/env python3
"""Compare full losses, scores, predictions and (when present) parameters."""
import argparse
import json
import math
from pathlib import Path

def compare(reference, candidate, atol=1e-6, rtol=1e-5):
    for key in ("epochs", "train_count", "test_count"):
        if reference[key] != candidate[key]:
            raise ValueError(f"Problemas diferentes: {key}")
    report = {}
    for key in ("epoch_losses", "scores", "parameters"):
        if key not in reference or key not in candidate:
            continue
        a, b = reference[key], candidate[key]
        if len(a) != len(b) or not a:
            raise ValueError(f"Dimensao invalida: {key}")
        max_error = 0.0
        for i, (x, y) in enumerate(zip(a, b)):
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(f"Valor nao finito: {key}[{i}]")
            max_error = max(max_error, abs(x-y))
            if abs(x-y) > atol + rtol*abs(x):
                raise ValueError(f"Divergencia {key}[{i}]: {x} != {y}; tolerancia {atol}+{rtol}*abs(ref)")
        report[key+"_max_abs"] = max_error
    if reference["predictions"] != candidate["predictions"]:
        raise ValueError("Previsoes diferentes")
    if reference["confusion"] != candidate["confusion"]:
        raise ValueError("Matrizes de confusao diferentes")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    try:
        result = compare(json.loads(args.reference.read_text()), json.loads(args.candidate.read_text()),
                         args.atol, args.rtol)
    except (ValueError, KeyError) as error:
        parser.exit(1, f"FALHOU: {error}\n")
    print(json.dumps({"status": "PASS", **result}, indent=2))
