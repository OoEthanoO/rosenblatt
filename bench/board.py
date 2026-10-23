"""Compare single-image and tiled inference on an already programmed ULX3S.

    python bench/board.py --port COM5 --count 32
    python bench/board.py --count 32       # emulator: correctness and event counts only

Both strategies check every hidden activation and output score. The tiled
path streams through the batch size that fits the device's remaining memory.
GPU cycles exclude the serial link; host inference time includes transfers,
kernel uploads, launches, polling and counter reads, but not NumPy checks.
This script does not program the board or change its flash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "sw"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ml import mnist  # noqa: E402
from ml.gpu import BatchMLP, GpuMLP  # noqa: E402
from ml.qmlp import ARTIFACTS, QuantMLP, preprocess  # noqa: E402
from rosenblatt.device import Device, EmuBackend  # noqa: E402
from rosenblatt.emu import Config  # noqa: E402
from rosenblatt.uart import UartBackend  # noqa: E402
from fpga.build import source_sha256  # noqa: E402


def array_digest(array: np.ndarray) -> str:
    """Hash a canonical int64 representation, including shape."""
    array = np.asarray(array)
    h = hashlib.sha256(json.dumps(list(array.shape)).encode())
    h.update(array.astype("<i8").tobytes())
    return h.hexdigest()


def verify_layer(actual: np.ndarray, expected: np.ndarray, name: str, start: int) -> None:
    if actual.shape != expected.shape:
        raise RuntimeError(f"{name}: shape {actual.shape}, expected {expected.shape}")
    mismatches = np.argwhere(actual != expected)
    if len(mismatches):
        row, col = map(int, mismatches[0])
        raise RuntimeError(f"image {start + row}: {name}[{col}] is {actual[row, col]}, "
                           f"expected {expected[row, col]}")


def measure_strategy(dev: Device, model: QuantMLP, inputs: np.ndarray, labels: np.ndarray,
                     name: str, result: dict, *, batch: int | None = None,
                     dp4a: bool | None = None, clock_hz: float | None = None,
                     start_index: int = 0, progress=None) -> None:
    """Populate a strategy's record as it runs, preserving verified chunks on failure."""
    dev.reset_heap()
    started = time.perf_counter()
    gpu = (GpuMLP(dev, model, dp4a=dp4a) if name == "per_image"
           else BatchMLP(dev, model, batch=batch, dp4a=dp4a))
    result.update(setup_wall_seconds=time.perf_counter() - started,
                  allocated_bytes=dev.memory_used, batch_capacity=1 if name == "per_image" else gpu.batch,
                  dp4a=gpu.dp4a, verified_images=0, correct=0, padded_images=0,
                  inference_wall_seconds=0.0, verification_wall_seconds=0.0, chunks=[])
    outputs = ((gpu.run(x) for x in inputs) if name == "per_image" else gpu.run_batches(inputs))
    processed = 0
    every = max(8, len(inputs) // 10)
    while processed < len(inputs):
        started = time.perf_counter()
        output = next(outputs)
        wall = time.perf_counter() - started
        hidden = output.hidden[None, :] if name == "per_image" else output.hidden
        logits = output.logits[None, :] if name == "per_image" else output.logits
        n = len(logits)
        chunk = inputs[processed:processed + n]
        checked = time.perf_counter()
        verify_layer(hidden, model.hidden(chunk), "hidden", start_index + processed)
        verify_layer(logits, model.logits(chunk), "logits", start_index + processed)
        predictions = np.argmax(logits, axis=1)
        correct = int((predictions == labels[processed:processed + n]).sum())
        result["verification_wall_seconds"] += time.perf_counter() - checked
        padded = 1 if name == "per_image" else output.padded_count
        result["chunks"].append({
            "start_index": start_index + processed, "count": n, "padded_count": padded,
            "predictions": predictions.tolist(), "correct": correct,
            "inference_wall_seconds": wall, "launches": [asdict(run) for run in output.runs],
        })
        result["inference_wall_seconds"] += wall
        result["verified_images"] += n
        result["padded_images"] += padded
        result["correct"] += correct
        previous = processed
        processed += n
        if progress and (processed // every != previous // every or processed == len(inputs)):
            progress(f"{name}: {processed}/{len(inputs)} images verified")

    runs = [run for chunk in result["chunks"] for run in chunk["launches"]]
    cycles = [run["cycles"] for run in runs]
    total_cycles = sum(cycles) if all(c is not None for c in cycles) else None
    stats = {}
    for run in runs:
        for key, value in run["stats"].items():
            stats[key] = stats.get(key, 0) + int(value)
    result.update(
        stats=stats, kernel_cycles=total_cycles,
        cycles_per_image=None if total_cycles is None else total_cycles / len(inputs),
        kernel_seconds=None if total_cycles is None or clock_hz is None else total_cycles / clock_hz,
        status="passed",
    )


def run_benchmark(dev: Device, model: QuantMLP, inputs: np.ndarray, labels: np.ndarray, *,
                  report: dict | None = None, batch: int | None = None, dp4a: bool | None = None,
                  clock_hz: float | None = None, start_index: int = 0, progress=None) -> dict:
    """Compare both paths on one device, reusing memory between strategies.

    Discards existing device allocations. Without hardware cycle counters,
    timing fields stay null; emulator wall time is never called GPU time.
    """
    inputs, labels = np.asarray(inputs), np.asarray(labels)
    if (inputs.ndim != 2 or inputs.shape[1] != model.w1.shape[1] or len(inputs) == 0 or
            inputs.dtype.kind not in "iu" or np.any(inputs < 0) or np.any(inputs > 127)):
        raise ValueError("inputs must be a nonempty integer matrix with values in 0..127")
    if labels.shape != (len(inputs),) or labels.dtype.kind not in "iu":
        raise ValueError("labels must be one integer per input image")
    if clock_hz is not None and (not math.isfinite(clock_hz) or clock_hz <= 0):
        raise ValueError("clock_hz must be positive and finite")
    if report is None:
        report = {}
    report.update(config=asdict(dev.cfg), count=len(inputs), start_index=start_index,
                  clock_hz=clock_hz, input_sha256=array_digest(inputs), label_sha256=array_digest(labels),
                  model={**{name + "_sha256": array_digest(getattr(model, name))
                            for name in ("w1", "b1", "w2", "b2")}, "shift": int(model.sh1)},
                  strategies=[])
    for name in ("per_image", "tiled"):
        result = {"name": name, "status": "running"}
        report["strategies"].append(result)
        if progress:
            progress(f"{name}: uploading model and allocating buffers")
        try:
            measure_strategy(dev, model, inputs, labels, name, result, batch=batch, dp4a=dp4a,
                             clock_hz=clock_hz, start_index=start_index, progress=progress)
        except Exception as exc:
            result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
    single, tiled = report["strategies"]
    report["kernel_speedup"] = (single["kernel_cycles"] / tiled["kernel_cycles"]
                                 if single["kernel_cycles"] is not None and tiled["kernel_cycles"] else None)
    return report


def save_report(path: Path, report: dict) -> None:
    """Replace a report atomically, including failed or interrupted runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as f:
        temp = Path(f.name)
        try:
            json.dump(report, f, indent=2, allow_nan=False)
            f.write("\n")
        except BaseException:
            f.close()
            temp.unlink(missing_ok=True)
            raise
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="already programmed serial GPU; omit for a 64 KiB emulator")
    ap.add_argument("--baudrate", type=int, default=115200)
    ap.add_argument("--timeout", type=float, default=2.0, help="serial transaction timeout in seconds")
    ap.add_argument("--clock-hz", type=float, default=25_000_000,
                    help="assumed FPGA clock for converting counters to seconds (default: 25000000)")
    ap.add_argument("--count", type=int, default=32)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--batch", type=int, help="tiled capacity, rounded up to 8; default: largest that fits")
    ap.add_argument("--scalar", action="store_true", help="disable DP4A in both strategies")
    ap.add_argument("--model", type=Path, default=ARTIFACTS / "mlp_int8.npz")
    ap.add_argument("--output", type=Path, default=ROOT / "build" / "bench" / "board-latest.json")
    a = ap.parse_args(argv)
    if a.count <= 0 or a.start < 0 or (a.batch is not None and a.batch <= 0):
        ap.error("--count and --batch must be positive; --start must be nonnegative")
    if (a.baudrate <= 0 or not math.isfinite(a.timeout) or a.timeout <= 0 or
            not math.isfinite(a.clock_hz) or a.clock_hz <= 0):
        ap.error("--baudrate, --timeout and --clock-hz must be positive and finite")
    report = {
        "schema_version": 1, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "serial" if a.port else "emulator", "port": a.port,
        "baudrate": a.baudrate if a.port else None,
        "clock_source": "user-specified conversion clock; not measured" if a.port else None,
        "sources": {name: source_sha256(ROOT / name) for name in (
            "bench/board.py", "ml/gpu.py", "ml/qmlp.py", "kernels/dense_int8.s", "kernels/dense_tiled.s",
            "sw/rosenblatt/device.py", "sw/rosenblatt/uart.py", "sw/rosenblatt/emu.py")},
    }
    # Clear any previous success before touching hardware. A failed attempt
    # must not leave a passing result at the requested destination.
    save_report(a.output, report)
    started = time.perf_counter()
    code = 0
    try:
        model = QuantMLP.load(a.model)
        report["model_file_sha256"] = hashlib.sha256(a.model.read_bytes()).hexdigest()
        data = mnist.load("test")
        if a.start + a.count > len(data["test_labels"]):
            raise ValueError("requested range exceeds the MNIST test set")
        selected = slice(a.start, a.start + a.count)
        inputs = preprocess(data["test_images"][selected])
        labels = data["test_labels"][selected]
        print(f"Rosenblatt benchmark: {report['backend']}" + (f" ({a.port})" if a.port else " (64 KiB)"), flush=True)
        connection = (UartBackend(a.port, baudrate=a.baudrate, timeout=a.timeout) if a.port
                      else nullcontext(EmuBackend(Config(gmem_bytes=64 * 1024))))
        with connection as backend:
            run_benchmark(Device(backend), model, inputs, labels, report=report,
                          batch=a.batch, dp4a=False if a.scalar else None,
                          clock_hz=a.clock_hz if a.port else None, start_index=a.start,
                          progress=lambda message: print(message, flush=True))
        report["status"] = "passed"
        for result in report["strategies"]:
            print(f"{result['name']}: {result['verified_images']} bit-exact images, "
                  f"{result['correct']}/{a.count} recognized; {result['allocated_bytes']:,} device bytes")
            print(f"  model setup {result['setup_wall_seconds']:.3f}s; "
                  f"host inference {result['inference_wall_seconds']:.3f}s; "
                  f"NumPy verification {result['verification_wall_seconds']:.3f}s")
            if result["kernel_cycles"] is not None:
                print(f"  GPU: {result['cycles_per_image']:,.1f} cycles/image, "
                      f"{result['kernel_seconds'] / a.count * 1e6:.1f} us/image at the assumed {a.clock_hz:g} Hz")
        if report["kernel_speedup"] is not None:
            print(f"Tiled kernel speedup: {report['kernel_speedup']:.2f}x (serial transfers excluded)")
        else:
            print("Emulator has no cycle timing; use --port for FPGA performance measurements.")
    except KeyboardInterrupt:
        report.update(status="interrupted", error="KeyboardInterrupt")
        code = 130
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print(f"FAIL: {exc}", file=sys.stderr)
        code = 1
    finally:
        report.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                      wall_seconds=time.perf_counter() - started)
        save_report(a.output, report)
        print(f"Report: {a.output.resolve()}", flush=True)
    if code and a.port:
        print("Press FIRE1 and restart the benchmark before using this serial connection again.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
