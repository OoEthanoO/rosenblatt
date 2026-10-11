"""Recognize MNIST digits using the GPU's instructions, without PyTorch.

    python -m ml.demo --count 10
    python -m ml.demo --count 10 --scalar
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from rosenblatt.device import Device, EmuBackend

from . import mnist
from .gpu import GpuMLP
from .qmlp import ARTIFACTS, QuantMLP, preprocess


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10, help="number of test images (default: 10)")
    parser.add_argument("--start", type=int, default=0, help="first MNIST test image index")
    parser.add_argument("--scalar", action="store_true", help="use scalar int8 multiply-adds instead of DP4A")
    parser.add_argument("--model", type=Path, default=ARTIFACTS / "mlp_int8.npz")
    args = parser.parse_args(argv)
    if args.count <= 0 or args.start < 0:
        parser.error("--count must be positive and --start must be nonnegative")
    if not args.model.is_file():
        parser.error(f"model not found: {args.model}; train one with python ml/train.py")

    model = QuantMLP.load(args.model)
    data = mnist.load("test")
    if args.start + args.count > len(data["test_labels"]):
        parser.error("requested range exceeds the MNIST test set")
    gpu = GpuMLP(Device(EmuBackend()), model, dp4a=not args.scalar)
    print(f"Rosenblatt emulator: {'scalar int8' if args.scalar else 'packed int8 DP4A'}")
    print("image  label  prediction  correct")
    correct = instructions = requests = 0
    started = time.perf_counter()
    for i in range(args.start, args.start + args.count):
        q0 = preprocess(data["test_images"][i])
        result = gpu.run(q0)
        np.testing.assert_array_equal(result.hidden, model.hidden(q0), err_msg=f"image {i}: hidden layer")
        np.testing.assert_array_equal(result.logits, model.logits(q0), err_msg=f"image {i}: logits")
        label = int(data["test_labels"][i])
        match = result.prediction == label
        correct += match
        instructions += sum(run.stats["instructions"] for run in result.runs)
        requests += sum(run.stats["gmem_reqs"] for run in result.runs)
        print(f"{i:5d}  {label:5d}  {result.prediction:10d}  {'yes' if match else 'no'}")
    print(f"Exact NumPy matches: {args.count}/{args.count} (hidden activations and all logits)")
    print(f"Sample accuracy: {correct}/{args.count} ({correct / args.count:.1%})")
    print(f"Warp instructions: {instructions:,}; global-memory requests: {requests:,}")
    print(f"Emulator wall time: {time.perf_counter() - started:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
