"""Profile MNIST inference on the cycle-accurate RTL (Icarus Verilog).

    python bench/profile_mnist.py              # every experiment, several in parallel
    python bench/profile_mnist.py --only per_image_dp4a tiled_dp4a
    python bench/profile_mnist.py --list

Each experiment builds the GPU with some parameters, runs one inference
strategy on real MNIST test images, checks every logit against the NumPy
reference, and saves cycle counts plus the hardware performance counters to
bench/results/<name>.json. bench/report.py turns those into tables and charts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "sw", ROOT / "tests", ROOT / "tests" / "rtl"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from rosenblatt.emu import Config  # noqa: E402

import simrun  # noqa: E402

RESULTS = ROOT / "bench" / "results"
PER_IMAGE_COUNT = 4   # the per-image kernels cost the same for every image
TILED_COUNT = 32      # 4 groups of 8 images x 2 neuron groups = 8 blocks


Spec = tuple[dict, str, bool, int, str]   # Config overrides, workload, dp4a, images, weight layout


def experiments() -> dict[str, Spec]:
    exp: dict[str, Spec] = {}
    # 1. four strategies on the default GPU (2 cores x 8 lanes x 4 warps)
    for workload, n in (("per_image", PER_IMAGE_COUNT), ("tiled", TILED_COUNT)):
        for dp4a in (False, True):
            exp[f"{workload}_{'dp4a' if dp4a else 'scalar'}"] = ({}, workload, dp4a, n, "rows")
    # 2-4. how each DP4A strategy reacts to the memory system and to more cores
    for workload, n in (("per_image", PER_IMAGE_COUNT), ("tiled", TILED_COUNT)):
        for lat in (1, 4, 16, 32):
            exp[f"latency{lat}_{workload}"] = ({"gmem_latency": lat}, workload, True, n, "rows")
        for interval in (2, 4):
            exp[f"interval{interval}_{workload}"] = ({"gmem_interval": interval}, workload, True, n, "rows")
        for cores in (1, 4):
            exp[f"cores{cores}_{workload}"] = ({"num_cores": cores}, workload, True, n, "rows")
    # 5. fewer warps to hide latency (the tiled kernel needs all four)
    exp["warps2_per_image"] = ({"num_warps": 2}, "per_image", True, PER_IMAGE_COUNT, "rows")
    # 6. load/store unit design: one instruction in flight (the first design),
    #    and parking warps instead of letting them retry while it is busy
    for workload, n in (("per_image", PER_IMAGE_COUNT), ("tiled", TILED_COUNT)):
        exp[f"lsu1_{workload}"] = ({"lsu_depth": 1}, workload, True, n, "rows")
        exp[f"lsu1_park_{workload}"] = ({"lsu_depth": 1, "park": True}, workload, True, n, "rows")
        exp[f"park_{workload}"] = ({"park": True}, workload, True, n, "rows")
    # 7. wider memory lines (one request moves 16 or 32 bytes), with the weights
    #    row-major or interleaved so that a warp's lanes read neighbouring words
    for workload, n in (("per_image", PER_IMAGE_COUNT), ("tiled", TILED_COUNT)):
        exp[f"interleaved_{workload}"] = ({}, workload, True, n, "interleaved")
        for line in (16, 32):
            for layout in ("rows", "interleaved"):
                exp[f"line{line}_{layout}_{workload}"] = ({"gmem_line_bytes": line}, workload, True, n, layout)
    return exp


def run_one(name: str, spec: Spec) -> dict:
    overrides, workload, dp4a, count, layout = spec
    cfg = replace(Config(), **overrides)
    out = RESULTS / f"{name}.json"
    out.unlink(missing_ok=True)
    simrun.run(cfg, "cocotb_bench", {
        "RB_WORKLOAD": workload, "RB_DP4A": "1" if dp4a else "0", "RB_COUNT": str(count),
        "RB_LAYOUT": layout, "RB_OUT": str(out),
    }, build_tag=name)
    if not out.exists():
        raise RuntimeError(f"{name}: simulation produced no result (see build/sim/*_{name})")
    return json.loads(out.read_text())


def main() -> int:
    exp = experiments()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="*", help="experiment names")
    p.add_argument("--jobs", type=int, default=6, help="simulations in parallel")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    if a.list:
        for k, (ov, w, d, n, layout) in exp.items():
            print(f"{k:28s} {w:9s} dp4a={d!s:5} images={n:2d} {layout:11s} {ov}")
        return 0
    names = a.only or list(exp)
    unknown = [n for n in names if n not in exp]
    if unknown:
        p.error(f"unknown experiments: {unknown}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    start = time.time()
    failures = []
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futures = {pool.submit(run_one, n, exp[n]): n for n in names}
        for fut in as_completed(futures):
            n = futures[fut]
            try:
                r = fut.result()
                print(f"[{time.time() - start:6.0f}s] {n:28s} {r['cycles'] / r['count']:9.0f} cycles/image  "
                      f"{r['correct']}/{r['count']} correct", flush=True)
            except Exception as e:  # keep going; report at the end
                failures.append(n)
                print(f"[{time.time() - start:6.0f}s] {n:28s} FAILED: {e}", flush=True)
    if failures:
        print("failed:", " ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
