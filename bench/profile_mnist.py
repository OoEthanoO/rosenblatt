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


Spec = tuple   # (Config overrides, workload, dp4a, images, weight layout[, kernel options])


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
    # 5. fewer warps to hide latency (the tiled kernel needs all four); with the
    #    scoreboard each warp hides some of its own latency
    exp["warps2_per_image"] = ({"num_warps": 2}, "per_image", True, PER_IMAGE_COUNT, "rows")
    exp["scoreboard_warps2_per_image"] = ({"num_warps": 2, "scoreboard": True}, "per_image", True,
                                          PER_IMAGE_COUNT, "rows")
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
    # 8. a register scoreboard: a global load stalls only the instructions that
    #    use its register, so one warp can have several loads in flight
    for workload, n in (("per_image", PER_IMAGE_COUNT), ("tiled", TILED_COUNT)):
        exp[f"scoreboard_{workload}"] = ({"scoreboard": True}, workload, True, n, "rows")
        exp[f"scoreboard_line32_{workload}"] = ({"scoreboard": True, "gmem_line_bytes": 32},
                                                workload, True, n, "interleaved")
    # 9. the one-image loop unrolled by two: fewer instructions, and with the
    #    scoreboard four loads of one warp in flight
    exp["unroll2_line32_per_image"] = ({"gmem_line_bytes": 32}, "per_image", True, PER_IMAGE_COUNT,
                                       "interleaved", {"unroll": 2})
    exp["unroll2_scoreboard_line32_per_image"] = ({"scoreboard": True, "gmem_line_bytes": 32},
                                                  "per_image", True, PER_IMAGE_COUNT, "interleaved",
                                                  {"unroll": 2})
    # ...which then waits for the load/store unit's four-instruction queue
    exp["lsu8_unroll2_scoreboard_line32_per_image"] = (
        {"scoreboard": True, "gmem_line_bytes": 32, "lsu_depth": 8},
        "per_image", True, PER_IMAGE_COUNT, "interleaved", {"unroll": 2})
    # 10. dp4ab: each lane loads one image's input word and broadcasts it,
    #     instead of every lane loading every image's word from shared memory
    exp["bcast_tiled"] = ({}, "tiled", True, TILED_COUNT, "rows", {"bcast": True})
    exp["bcast_scoreboard_line32_tiled"] = ({"scoreboard": True, "gmem_line_bytes": 32}, "tiled", True,
                                            TILED_COUNT, "interleaved", {"bcast": True})
    # 11. double-buffered tiles: one barrier per stage instead of two
    exp["dbuf_bcast_tiled"] = ({}, "tiled", True, TILED_COUNT, "rows", {"bcast": True, "dbuf": True})
    exp["dbuf_bcast_scoreboard_line32_tiled"] = ({"scoreboard": True, "gmem_line_bytes": 32}, "tiled", True,
                                                 TILED_COUNT, "interleaved", {"bcast": True, "dbuf": True})
    return exp


def run_one(name: str, spec: Spec) -> dict:
    overrides, workload, dp4a, count, layout, *rest = spec
    opts = rest[0] if rest else {}
    cfg = replace(Config(), **overrides)
    out = RESULTS / f"{name}.json"
    out.unlink(missing_ok=True)
    simrun.run(cfg, "cocotb_bench", {
        "RB_WORKLOAD": workload, "RB_DP4A": "1" if dp4a else "0", "RB_COUNT": str(count),
        "RB_LAYOUT": layout, "RB_UNROLL": str(opts.get("unroll", 1)),
        "RB_BCAST": "1" if opts.get("bcast") else "0", "RB_DBUF": "1" if opts.get("dbuf") else "0",
        "RB_OUT": str(out),
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
        for k, (ov, w, d, n, layout, *rest) in exp.items():
            print(f"{k:40s} {w:9s} dp4a={d!s:5} images={n:2d} {layout:11s} {rest[0] if rest else {}} {ov}")
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
                print(f"[{time.time() - start:6.0f}s] {n:36s} {r['cycles'] / r['count']:9.0f} cycles/image  "
                      f"{r['correct']}/{r['count']} correct", flush=True)
            except Exception as e:  # keep going; report at the end
                failures.append(n)
                print(f"[{time.time() - start:6.0f}s] {n:36s} FAILED: {e}", flush=True)
    if failures:
        print("failed:", " ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
