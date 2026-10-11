"""Turn bench/results/*.json into tables (Markdown) and charts (PNG) for the docs.

    python bench/report.py        # writes bench/results/summary.md and docs/img/*.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results"
IMG = ROOT / "docs" / "img"


def load() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))}


def metrics(r: dict) -> dict:
    c, n, cfg = r["counters"], r["count"], r["config"]
    cyc = r["cycles"]
    return {
        "cycles_per_image": cyc / n,
        "instr_per_image": c["issued"] / n,
        "gmem_reqs_per_image": r["device_gmem_reqs"] / n,
        "smem_wavefronts_per_image": c["smem_cycles"] / n,
        "ipc_per_core": c["issued"] / max(1, c["active"]),
        "mem_port_busy": r["device_gmem_reqs"] * cfg["gmem_interval"] / cyc,
        "idle": c["idle"] / max(1, c["active"]),
        "idle_mem": c["idle_mem"] / max(1, c["active"]),
        "replays_per_image": c["replays"] / n,
        "lane_util": c["lane_ops"] / max(1, c["issued"] * cfg["num_lanes"]),
        "correct": f"{r['correct']}/{n}",
    }


def table(rows: list[tuple[str, dict]], cols: list[tuple[str, str, str]]) -> str:
    head = "| " + " | ".join(["Experiment"] + [c[1] for c in cols]) + " |"
    sep = "| --- " + "| ---: " * len(cols) + "|"
    out = [head, sep]
    for name, m in rows:
        out.append("| " + " | ".join([name] + [format(m[k], f) if f else str(m[k]) for k, _, f in cols]) + " |")
    return "\n".join(out)


COLS = [("cycles_per_image", "cycles / image", ",.0f"),
        ("instr_per_image", "warp instructions / image", ",.0f"),
        ("gmem_reqs_per_image", "global requests / image", ",.0f"),
        ("ipc_per_core", "IPC per core", ".2f"),
        ("mem_port_busy", "memory port busy", ".0%"),
        ("idle_mem", "issue slots lost to memory waits", ".0%"),
        ("replays_per_image", "replays / image", ",.0f"),
        ("correct", "correct", "")]


def main() -> int:
    res = load()
    if not res:
        print("no results: run bench/profile_mnist.py first")
        return 1
    m = {k: metrics(v) for k, v in res.items()}
    md = ["# Profiling results", "",
          "Cycle-accurate RTL simulation (Icarus Verilog) of MNIST inference, checked "
          "logit-for-logit against the NumPy reference. Default GPU: 2 cores x 8 lanes x "
          "4 warps, global memory latency 8 cycles, one 32-bit request per cycle.", ""]

    def section(title: str, names: list[str], note: str = "") -> None:
        rows = [(n, m[n]) for n in names if n in m]
        if rows:
            md.extend([f"## {title}", ""] + ([note, ""] if note else []) + [table(rows, COLS), ""])

    section("Inference strategies", ["per_image_scalar", "per_image_dp4a", "tiled_scalar", "tiled_dp4a"])
    for w in ("per_image", "tiled"):
        default = f"{w}_dp4a"
        section(f"Memory latency ({w})", [f"latency{x}_{w}" for x in (1, 4)] + [default] +
                [f"latency{x}_{w}" for x in (16, 32)], "Default latency is 8 cycles.")
        section(f"Memory bandwidth ({w})", [default] + [f"interval{x}_{w}" for x in (2, 4)],
                "intervalN: the memory accepts one request every N cycles.")
        section(f"Cores ({w})", [f"cores1_{w}", default, f"cores4_{w}"])
        section(f"Load/store unit design ({w})",
                [f"lsu1_{w}", f"lsu1_park_{w}", default, f"park_{w}"],
                "lsu1: one global memory instruction in flight per core (the first design); "
                "park: warps sleep while the LSU is busy instead of retrying.")
    section("Warps per core (per_image)", ["warps2_per_image", "per_image_dp4a"])
    (RESULTS / "summary.md").write_text("\n".join(md) + "\n")
    print((RESULTS / "summary.md").read_text())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping charts")
        return 0
    IMG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    blue, orange, grey = "#2563eb", "#ea580c", "#9ca3af"

    # 1. strategies
    names = [("per_image_scalar", "per image\nscalar"), ("per_image_dp4a", "per image\nDP4A"),
             ("tiled_scalar", "tiled + shared\nscalar"), ("tiled_dp4a", "tiled + shared\nDP4A")]
    names = [(k, lab) for k, lab in names if k in m]
    if names:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        xs = range(len(names))
        ax[0].bar(xs, [m[k]["cycles_per_image"] for k, _ in names], color=[grey, grey, blue, blue])
        ax[0].set_xticks(list(xs), [lab for _, lab in names])
        ax[0].set_ylabel("cycles per image")
        ax[0].set_title("Time per digit")
        ax[1].bar(xs, [m[k]["gmem_reqs_per_image"] for k, _ in names], color=[grey, grey, orange, orange])
        ax[1].set_xticks(list(xs), [lab for _, lab in names])
        ax[1].set_ylabel("global memory requests per image")
        ax[1].set_title("Data moved from global memory")
        fig.tight_layout()
        fig.savefig(IMG / "strategies.png", dpi=150)
        plt.close(fig)

    # 2. latency and bandwidth sensitivity
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for w, col, lab in (("per_image", grey, "per image"), ("tiled", blue, "tiled + shared")):
        pts = [(1, f"latency1_{w}"), (4, f"latency4_{w}"), (8, f"{w}_dp4a"), (16, f"latency16_{w}"),
               (32, f"latency32_{w}")]
        pts = [(x, m[k]["cycles_per_image"]) for x, k in pts if k in m]
        if pts:
            ax[0].plot(*zip(*pts), marker="o", color=col, label=lab)
        pts = [(1, f"{w}_dp4a"), (2, f"interval2_{w}"), (4, f"interval4_{w}")]
        pts = [(4 / x, m[k]["cycles_per_image"]) for x, k in pts if k in m]
        if pts:
            ax[1].plot(*zip(*pts), marker="o", color=col, label=lab)
    ax[0].set_xlabel("global memory latency (cycles)")
    ax[0].set_ylabel("cycles per image")
    ax[0].set_title("Latency")
    ax[0].legend(frameon=False)
    ax[1].set_xlabel("global memory bandwidth (bytes / cycle)")
    ax[1].set_ylabel("cycles per image")
    ax[1].set_title("Bandwidth")
    ax[1].set_xscale("log", base=2)
    ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMG / "memory_sensitivity.png", dpi=150)
    plt.close(fig)

    # 3. scaling with cores
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for w, col, lab in (("per_image", grey, "per image"), ("tiled", blue, "tiled + shared")):
        pts = [(1, f"cores1_{w}"), (2, f"{w}_dp4a"), (4, f"cores4_{w}")]
        pts = [(c, m[k]["cycles_per_image"]) for c, k in pts if k in m]
        if len(pts) > 1:
            base = pts[0][1]
            ax.plot([c for c, _ in pts], [base / v for _, v in pts], marker="o", color=col, label=lab)
    ax.plot([1, 4], [1, 4], ls=":", color="#6b7280", label="ideal")
    ax.set_xlabel("cores")
    ax.set_ylabel("speedup over one core")
    ax.set_title("Adding cores")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMG / "core_scaling.png", dpi=150)
    plt.close(fig)
    print(f"charts written to {IMG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
