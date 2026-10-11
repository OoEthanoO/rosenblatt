"""Build the GPU for an FPGA with the open-source flow (Yosys + nextpnr).

    python fpga/build.py ulx3s-85f              # bitstream + resource/timing report
    python fpga/build.py ulx3s-25f --synth-only # just count resources
    python fpga/build.py --list

Outputs go to build/fpga/<target>/: top.bit (load it with
`openFPGALoader -b ulx3s build/fpga/<target>/top.bit`), report.json from
nextpnr, and summary.json with the numbers quoted in docs/fpga.md.
The tools come from the OSS CAD Suite (found on PATH, in ~/tools/oss-cad-suite,
or in $OSS_CAD_SUITE).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl"
RTL_FILES = ["rb_gpu.v", "rb_core.v", "rb_lsu.v", "rb_smem.v", "rb_alu.v", "rb_regfile.v",
             "rb_gmem.v", "rb_uart_rx.v", "rb_uart_tx.v", "rb_uart_bridge.v", "rb_serial_gpu.v"]

# ULX3S variants (all CABGA381). The bigger the FPGA, the more of the GPU fits.
TARGETS = {
    "ulx3s-85f": {"device": "85k", "params": {}},   # 2 cores x 8 lanes, DP4A
    "ulx3s-45f": {"device": "45k", "params": {"NUM_CORES": 1}},
    "ulx3s-25f": {"device": "25k", "params": {"NUM_CORES": 1, "NUM_LANES": 4, "MUL_WIDTH": 16}},
    "ulx3s-12f": {"device": "12k", "params": {"NUM_CORES": 1, "NUM_LANES": 4, "NUM_WARPS": 2,
                                               "GMEM_AW": 13, "MUL_WIDTH": 16, "ENABLE_DP4A": 0}},
}
BOARD = {"top": ROOT / "fpga" / "ulx3s" / "top.v", "lpf": ROOT / "fpga" / "ulx3s" / "ulx3s.lpf",
         "package": "CABGA381", "freq_mhz": 25}


def find_tools() -> None:
    if shutil.which("yosys"):
        return
    base = Path(os.environ.get("OSS_CAD_SUITE", Path.home() / "tools" / "oss-cad-suite"))
    if not (base / "bin").is_dir():
        sys.exit("yosys not found: install the OSS CAD Suite or set OSS_CAD_SUITE")
    os.environ["PATH"] = os.pathsep.join([str(base / "bin"), str(base / "lib"), os.environ["PATH"]])


def run(cmd: list[str], log: Path) -> float:
    t = time.time()
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    if p.returncode != 0:
        tail = log.read_text(errors="replace").splitlines()[-25:]
        sys.exit(f"{cmd[0]} failed (see {log}):\n" + "\n".join(tail))
    return time.time() - t


def build(name: str, synth_only: bool = False, seed: int = 1) -> dict:
    target = TARGETS[name]
    out = ROOT / "build" / "fpga" / name
    out.mkdir(parents=True, exist_ok=True)
    files = " ".join((RTL / f).relative_to(ROOT).as_posix() for f in RTL_FILES)
    chparam = "".join(f" -set {k} {v}" for k, v in target["params"].items())
    script = [f"read_verilog -defer -I{RTL.relative_to(ROOT).as_posix()} {files} "
              f"{BOARD['top'].relative_to(ROOT).as_posix()}"]
    if chparam:
        script.append(f"chparam{chparam} top")
    script += [f"synth_ecp5 -top top -json {(out / 'top.json').relative_to(ROOT).as_posix()}",
               f"tee -o {(out / 'synth_stat.txt').relative_to(ROOT).as_posix()} stat"]
    (out / "synth.ys").write_text("\n".join(script) + "\n")
    summary: dict = {"target": name, "params": target["params"]}
    summary["synth_seconds"] = round(run(["yosys", "-q", "-l", str(out / "yosys.log"), str(out / "synth.ys")],
                                         out / "yosys.out"), 1)
    stat = (out / "synth_stat.txt").read_text()
    summary["cells"] = {k: int(v) for v, k in re.findall(r"^\s+(\d+)\s+([A-Z][A-Z0-9_]+)$", stat, re.M)}
    if synth_only:
        return summary
    summary["pnr_seconds"] = round(run([
        "nextpnr-ecp5", f"--{target['device']}", "--package", BOARD["package"],
        "--json", str(out / "top.json"), "--lpf", str(BOARD["lpf"]), "--textcfg", str(out / "top.config"),
        "--freq", str(BOARD["freq_mhz"]), "--report", str(out / "report.json"), "--seed", str(seed),
        "--timing-allow-fail",
    ], out / "nextpnr.log"), 1)
    report = json.loads((out / "report.json").read_text())
    summary["fmax_mhz"] = {clk: round(v["achieved"], 1) for clk, v in report.get("fmax", {}).items()}
    summary["utilization"] = {k: v for k, v in report.get("utilization", {}).items() if v.get("used")}
    run(["ecppack", "--compress", str(out / "top.config"), str(out / "top.bit")], out / "ecppack.log")
    summary["bitstream"] = str((out / "top.bit").relative_to(ROOT))
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("target", nargs="?", default="ulx3s-85f")
    p.add_argument("--synth-only", action="store_true")
    p.add_argument("--seed", type=int, default=1, help="placement seed for nextpnr")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    if a.list or a.target not in TARGETS:
        for k, v in TARGETS.items():
            print(f"{k:10s} ECP5-{v['device'].upper()}  {v['params'] or 'default configuration'}")
        return 0 if a.list else 1
    find_tools()
    summary = build(a.target, a.synth_only, a.seed)
    path = ROOT / "build" / "fpga" / a.target / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
