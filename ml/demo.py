"""Recognize MNIST digits using the GPU's instructions, without PyTorch.

    python -m ml.demo --count 10
    python -m ml.demo --count 10 --scalar
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from rosenblatt.device import Device, EmuBackend
from rosenblatt.uart import UartBackend

from . import mnist
from .gpu import GpuMLP
from .qmlp import ARTIFACTS, QuantMLP, preprocess


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10, help="number of test images (default: 10)")
    parser.add_argument("--start", type=int, default=0, help="first MNIST test image index")
    parser.add_argument("--scalar", action="store_true", help="use scalar int8 multiply-adds instead of DP4A")
    parser.add_argument("--port", help="serial GPU port, e.g. COM3 or /dev/ttyUSB0 (default: emulator)")
    parser.add_argument("--baudrate", type=int, default=115200, help="serial baud rate (default: 115200)")
    parser.add_argument("--timeout", type=float, default=2.0, help="serial transaction timeout in seconds")
    parser.add_argument("--model", type=Path, default=ARTIFACTS / "mlp_int8.npz")
    args = parser.parse_args(argv)
    if args.count <= 0 or args.start < 0:
        parser.error("--count must be positive and --start must be nonnegative")
    if args.baudrate <= 0 or not 0 < args.timeout < float("inf"):
        parser.error("--baudrate and --timeout must be positive and finite")
    if not args.model.is_file():
        parser.error(f"model not found: {args.model}; train one with python ml/train.py")

    model = QuantMLP.load(args.model)
    data = mnist.load("test")
    if args.start + args.count > len(data["test_labels"]):
        parser.error("requested range exceeds the MNIST test set")
    connection = (UartBackend(args.port, baudrate=args.baudrate, timeout=args.timeout)
                  if args.port else nullcontext(EmuBackend()))
    with connection as backend:
        gpu = GpuMLP(Device(backend), model, dp4a=False if args.scalar else None)
        label = f"serial GPU ({args.port})" if args.port else "emulator"
        print(f"Rosenblatt {label}: {'packed int8 DP4A' if gpu.dp4a else 'scalar int8'}")
        return run_demo(gpu, model, data, args.start, args.count, serial=bool(args.port))


def run_demo(gpu: GpuMLP, model: QuantMLP, data: dict[str, np.ndarray],
             start: int, count: int, *, serial: bool = False) -> int:
    """Run and verify a selected test-set range on an initialized device."""
    print("image  label  prediction  correct")
    correct = instructions = requests = cycles = 0
    started = time.perf_counter()
    for i in range(start, start + count):
        q0 = preprocess(data["test_images"][i])
        result = gpu.run(q0)
        np.testing.assert_array_equal(result.hidden, model.hidden(q0), err_msg=f"image {i}: hidden layer")
        np.testing.assert_array_equal(result.logits, model.logits(q0), err_msg=f"image {i}: logits")
        label = int(data["test_labels"][i])
        match = result.prediction == label
        correct += match
        instructions += sum(run.stats["instructions"] for run in result.runs)
        requests += sum(run.stats["gmem_reqs"] for run in result.runs)
        cycles += sum(run.cycles or 0 for run in result.runs)
        print(f"{i:5d}  {label:5d}  {result.prediction:10d}  {'yes' if match else 'no'}")
    print(f"Exact NumPy matches: {count}/{count} (hidden activations and all logits)")
    print(f"Sample accuracy: {correct}/{count} ({correct / count:.1%})")
    print(f"Warp instructions: {instructions:,}; global-memory requests: {requests:,}")
    if serial:
        print(f"GPU kernel cycles: {cycles:,} (excludes serial transfers)")
    print(f"{'Host' if serial else 'Emulator'} wall time: {time.perf_counter() - started:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
