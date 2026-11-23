#!/usr/bin/env python3
"""Create a transfer archive without Git metadata, executables or credentials."""
import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument("output",type=Path)
args=p.parse_args()
target=args.output.resolve()
with ZipFile(target,"w",compression=ZIP_DEFLATED,compresslevel=6) as z:
    for f in sorted(root.rglob("*")):
        rel=f.relative_to(root)
        if (not f.is_file() or f.resolve()==target or
            any(part in (".git","build","__pycache__",".venv") for part in rel.parts) or
            str(rel).replace("\\","/").startswith("logs/") or
            f.name in ("id_rsa","id_ed25519",".env") or f.suffix in (".pem",".key") or
            str(rel).replace("\\","/").startswith("results/verification/") or
            (str(rel).replace("\\","/").startswith("results/cluster/") and
             not str(rel).replace("\\","/").startswith(("results/cluster/minimo/","results/cluster/logs/")))):
            continue
        name = str(rel).replace("\\","/")
        if f.suffix in (".sh", ".slurm"):
            z.writestr(name, f.read_bytes().replace(b"\r\n", b"\n"))
        else:
            z.write(f,name)
print(target)
