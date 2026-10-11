"""Run the digit network on the NVIDIA GPU and check it bit for bit.

    python cuda/run.py            # full MNIST test set (10,000 images)

Exports the int8 model and the test images, compiles mnist_bench.cu with nvcc
(on Windows inside the Visual Studio build environment), runs it, compares
every logit of every method with the NumPy reference in ml/qmlp.py, and saves
cuda/results.json.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml import mnist  # noqa: E402
from ml.qmlp import QuantMLP, preprocess  # noqa: E402

BUILD = ROOT / "build" / "cuda"
SRC = ROOT / "cuda" / "mnist_bench.cu"


def find_vcvars() -> str | None:
    pats = [r"C:\Program Files*\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat"]
    hits = sorted(h for p in pats for h in glob.glob(p))
    return hits[-1] if hits else None


def compile_bench() -> Path:
    exe = BUILD / ("mnist_bench.exe" if os.name == "nt" else "mnist_bench")
    if exe.exists() and exe.stat().st_mtime > SRC.stat().st_mtime:
        return exe
    nvcc = shutil.which("nvcc") or next(iter(sorted(glob.glob(
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\nvcc.exe"))), None)
    if not nvcc:
        sys.exit("nvcc not found: install the CUDA toolkit")
    cmd = f'"{nvcc}" -O3 -std=c++17 -arch=native -o "{exe}" "{SRC}" -lcublas'
    if os.name == "nt":
        vcvars = find_vcvars()
        if not vcvars:
            sys.exit("vcvars64.bat not found: install Visual Studio Build Tools (C++)")
        # shell=True passes the string to cmd.exe unchanged (a list would re-quote it)
        p = subprocess.run(f'call "{vcvars}" >nul && {cmd}', shell=True, capture_output=True, text=True)
    else:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit("nvcc failed:\n" + p.stdout[-3000:] + p.stderr[-3000:])
    return exe


def main() -> int:
    BUILD.mkdir(parents=True, exist_ok=True)
    model = QuantMLP.load()
    data = mnist.load("test")
    q0 = preprocess(data["test_images"]).astype(np.int8)
    labels = data["test_labels"]
    n_out_pad = 16
    dims = np.array([len(q0), q0.shape[1], model.w1.shape[0], model.w2.shape[0], n_out_pad, model.sh1],
                    dtype=np.int32)
    dims.tofile(BUILD / "dims.bin")
    q0.tofile(BUILD / "x.bin")
    model.w1.astype(np.int8).tofile(BUILD / "w1.bin")
    model.b1.astype(np.int32).tofile(BUILD / "b1.bin")
    model.w2.astype(np.int8).tofile(BUILD / "w2.bin")
    model.b2.astype(np.int32).tofile(BUILD / "b2.bin")

    exe = compile_bench()
    p = subprocess.run([str(exe), str(BUILD)], capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"benchmark failed:\n{p.stdout}\n{p.stderr}")
    lines = [json.loads(line) for line in p.stdout.splitlines() if line.startswith("{")]
    header, methods = lines[0], lines[1:]

    reference = model.logits(q0.astype(np.int32))
    for m in methods:
        got = np.fromfile(BUILD / f"logits_{m['method']}.bin", dtype=np.int32).reshape(reference.shape)
        m["bit_exact"] = bool(np.array_equal(got, reference))
        m["accuracy"] = float((np.argmax(got, axis=1) == labels).mean())
        m["images_per_second"] = 1e9 / m["ns_per_image"]
        ops = 2 * (model.w1.size + model.w2.size)   # multiply + add per weight
        m["tops"] = ops * m["images_per_second"] / 1e12
        print(f"{m['method']:7s} {m['ns_per_image']:9.3f} ns/image  {m['images_per_second']:13,.0f} images/s  "
              f"{m['tops']:6.2f} TOPS  bit-exact={m['bit_exact']}  accuracy={m['accuracy']:.4f}")

    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.max.sm,clocks.max.mem,driver_version",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    result = {"gpu": header, "nvidia_smi": smi, "methods": methods,
              "ops_per_image": 2 * int(model.w1.size + model.w2.size)}
    (ROOT / "cuda" / "results.json").write_text(json.dumps(result, indent=2))
    print(f"{header['device']} ({header['sms']} SMs); image upload for the whole batch: {header['upload_ms']:.3f} ms")
    if not all(m["bit_exact"] for m in methods):
        print("ERROR: some method does not match the reference")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
