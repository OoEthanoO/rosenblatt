"""Rosenblatt versus the NVIDIA GPU, from the measured results.

    python bench/compare.py [--mhz 25]

Reads bench/results/tiled_dp4a.json and per_image_dp4a.json (cycle counts from
the RTL), cuda/results.json and cuda/results_x10.json (if present), and prints
Markdown tables used in docs/writeup.md: time per digit on both machines and
where the difference comes from.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mhz", type=float, default=25.0, help="Rosenblatt clock (the ULX3S oscillator: 25)")
    a = ap.parse_args()

    rb = {k: json.loads((ROOT / "bench" / "results" / f"{k}.json").read_text())
          for k in ("per_image_dp4a", "tiled_dp4a")}
    cuda = json.loads((ROOT / "cuda" / "results.json").read_text())
    x10_path = ROOT / "cuda" / "results_x10.json"
    cuda10 = json.loads(x10_path.read_text()) if x10_path.exists() else None
    gpu = cuda["gpu"]
    ops = cuda["ops_per_image"]
    m = {r["method"]: r for r in cuda["methods"]}

    cfg = rb["tiled_dp4a"]["config"]
    lanes = cfg["num_cores"] * cfg["num_lanes"]
    f_rb = a.mhz * 1e6
    f_nv = cuda.get("clocks_during_run", {}).get("sm_mhz_load", 0) * 1e6
    nv_lanes = gpu["sms"] * 128          # 128 integer/FP32 lanes per Blackwell SM

    def rb_ns(name: str) -> float:
        r = rb[name]
        return r["cycles"] / r["count"] / f_rb * 1e9

    print(f"## Time per digit (Rosenblatt at {a.mhz:g} MHz)\n")
    print("| Implementation | Rosenblatt | RTX 5080 | Ratio |")
    print("| --- | ---: | ---: | ---: |")
    pairs = [("one thread per neuron, DP4A, no shared memory", "per_image_dp4a", "dp4a"),
             ("shared-memory tiles + DP4A", "tiled_dp4a", "tiled")]
    for label, rk, nk in pairs:
        r, n = rb_ns(rk), m[nk]["ns_per_image"]
        print(f"| {label} | {r / 1000:,.1f} us | {n:.2f} ns | {r / n:,.0f}x |")
    print(f"| best available (cuBLAS, tensor cores) | {rb_ns('tiled_dp4a') / 1000:,.1f} us | "
          f"{m['cublas']['ns_per_image']:.2f} ns | {rb_ns('tiled_dp4a') / m['cublas']['ns_per_image']:,.0f}x |")
    if cuda10:
        m10 = {r["method"]: r for r in cuda10["methods"]}
        print(f"\nWith 100,000 images per launch: tiled {m10['tiled']['ns_per_image']:.2f} ns, "
              f"cuBLAS {m10['cublas']['ns_per_image']:.2f} ns per digit.")

    if not f_nv:
        print("\n(no measured SM clock in cuda/results.json: rerun cuda/run.py)")
        return 0
    rb_peak = lanes * 4 * 2 * f_rb                  # DP4A: 4 multiply-adds per lane per cycle
    nv_peak = nv_lanes * 4 * 2 * f_nv               # upper bound: one DP4A per lane per cycle
    rb_ops = ops / (rb_ns("tiled_dp4a") * 1e-9)
    nv_ops = ops / (m["tiled"]["ns_per_image"] * 1e-9)
    bw_rb = 4 * f_rb / cfg["gmem_interval"]
    bw_nv = gpu["mem_bus_bits"] / 8 * 2 * cuda["clocks_during_run"]["mem_mhz_max"] * 1e6
    print("\n## Where the difference comes from (tiled DP4A kernel on both)\n")
    print("| | Rosenblatt | RTX 5080 | Ratio |")
    print("| --- | ---: | ---: | ---: |")
    print(f"| clock | {a.mhz:g} MHz | {f_nv / 1e6:,.0f} MHz (measured) | {f_nv / f_rb:,.0f}x |")
    print(f"| lanes | {lanes} | {nv_lanes:,} | {nv_lanes / lanes:,.0f}x |")
    print(f"| DP4A peak (lanes x clock x 8 ops) | {rb_peak / 1e9:.1f} GOPS | {nv_peak / 1e12:.0f} TOPS | "
          f"{nv_peak / rb_peak:,.0f}x |")
    print(f"| achieved on this kernel | {rb_ops / 1e9:.2f} GOPS ({rb_ops / rb_peak:.0%} of peak) | "
          f"{nv_ops / 1e12:.1f} TOPS ({nv_ops / nv_peak:.0%} of peak) | {nv_ops / rb_ops:,.0f}x |")
    print(f"| memory bandwidth | {bw_rb / 1e6:,.0f} MB/s | {bw_nv / 1e9:,.0f} GB/s | {bw_nv / bw_rb:,.0f}x |")
    print(f"| on-chip memory | {cfg['num_cores'] * cfg['smem_bytes'] // 1024} KiB shared | "
          f"{gpu['sms'] * gpu['smem_per_sm'] // 1024:,} KiB shared + {gpu['l2_bytes'] // 2**20} MiB L2 | |")
    print(f"| threads resident | {cfg['num_cores'] * cfg['num_lanes'] * cfg['num_warps']} | "
          f"{gpu['sms'] * gpu['max_threads_per_sm']:,} | "
          f"{gpu['sms'] * gpu['max_threads_per_sm'] / (cfg['num_cores'] * cfg['num_lanes'] * cfg['num_warps']):,.0f}x |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
