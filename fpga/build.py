"""Build the GPU for an FPGA with the open-source flow (Yosys + nextpnr).

    python fpga/build.py ulx3s-85f              # bitstream + resource/timing report
    python fpga/build.py ulx3s-25f --synth-only # just count resources
    python fpga/build.py ulx3s-85f --synth-only --param GMEM_LINE=32   # a variant
    python fpga/build.py --list

Outputs go to build/fpga/<target>/: top.bit (load it with
`openFPGALoader -b ulx3s build/fpga/<target>/top.bit`), report.json from
nextpnr, and summary.json with the numbers quoted in docs/fpga.md.
The tools come from the OSS CAD Suite (found on PATH, in ~/tools/oss-cad-suite,
or in $OSS_CAD_SUITE).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl"
RTL_FILES = ["rb_gpu.v", "rb_core.v", "rb_lsu.v", "rb_smem.v", "rb_alu.v", "rb_regfile.v",
             "rb_gmem.v", "rb_uart_rx.v", "rb_uart_tx.v", "rb_uart_bridge.v", "rb_serial_gpu.v"]

# ULX3S variants (all CABGA381). The bigger the FPGA, the more of the GPU fits.
TARGETS = {
    "ulx3s-85f": {"device": "85k", "params": {}},   # 2 cores x 8 lanes, DP4A
    "ulx3s-45f": {"device": "45k", "params": {"NUM_CORES": 1}},
    "ulx3s-25f": {"device": "25k", "params": {"NUM_CORES": 1, "NUM_LANES": 4, "MUL_WIDTH": 16}},
}   # the 12F's 32 block RAMs cannot hold the 64 KiB global memory the digit network needs
BOARD = {"top": ROOT / "fpga" / "ulx3s" / "top.v", "lpf": ROOT / "fpga" / "ulx3s" / "ulx3s.lpf",
         "package": "CABGA381", "freq_mhz": 25, "speed": "6"}
DEFAULT_REVISION = "3.1.7"
REVISIONS = ("3.0", "3.1.7")


def find_tools(required=("yosys", "nextpnr-ecp5", "ecppack")) -> None:
    base = Path(os.environ.get("OSS_CAD_SUITE", Path.home() / "tools" / "oss-cad-suite"))
    if any(shutil.which(tool) is None for tool in required) and (base / "bin").is_dir():
        os.environ["PATH"] = os.pathsep.join([str(base / "bin"), str(base / "lib"),
                                             os.environ.get("PATH", "")])
    missing = [tool for tool in required if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f"{', '.join(missing)} not found: install the OSS CAD Suite or set OSS_CAD_SUITE")


def run(cmd: list[str], log: Path) -> float:
    t = time.time()
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    if p.returncode != 0:
        tail = log.read_text(errors="replace").splitlines()[-25:]
        raise RuntimeError(f"{cmd[0]} exited with {p.returncode} (see {log}):\n" + "\n".join(tail))
    return time.time() - t


def output_dir(name: str, extra: dict[str, int] | None = None,
               revision: str = DEFAULT_REVISION) -> Path:
    suffix = "" if revision == DEFAULT_REVISION else f"+pcb{revision}"
    suffix += "".join(f"+{k}={v}" for k, v in sorted((extra or {}).items()))
    return ROOT / "build" / "fpga" / (name + suffix)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_sha256(path: Path) -> str:
    """The checksum of a text source, with CRLF line endings read as LF: Git on
    Windows usually checks files out with CRLF, and a Windows and a Linux
    checkout of the same commit must agree."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def timing_passes(report: dict, required_mhz: float) -> bool:
    clocks = report.get("fmax", {})
    return bool(clocks) and all(
        values.get("constraint", 0) >= required_mhz
        and values.get("achieved", 0) >= values["constraint"]
        for values in clocks.values()
    )


def build(name: str, synth_only: bool = False, seed: int = 1, router: str = "router1",
          extra: dict[str, int] | None = None, revision: str = DEFAULT_REVISION) -> dict:
    """extra: parameters of top.v on top of the target's own; the build then goes
    to its own directory, e.g. build/fpga/ulx3s-85f+GMEM_LINE=32."""
    target = dict(TARGETS[name])
    target["params"] = {**target["params"], **(extra or {})}
    defaults = {k: int(v.replace("_", "")) for k, v in re.findall(
        r"parameter\s+(\w+)\s*=\s*([\d_]+)", BOARD["top"].read_text())}
    unknown = set(target["params"]) - defaults.keys()
    if unknown:
        raise ValueError(f"unknown top-level parameters: {', '.join(sorted(unknown))}")
    effective = {**defaults, **target["params"]}
    if effective["CLK_HZ"] != BOARD["freq_mhz"] * 1_000_000:
        raise ValueError("the ULX3S wrapper uses the fixed 25 MHz oscillator; CLK_HZ must be 25000000")
    if effective["BAUD"] <= 0 or effective["BAUD"] > effective["CLK_HZ"] // 4:
        raise ValueError("BAUD must be positive and allow at least four clocks per bit")
    if revision not in REVISIONS:
        raise ValueError(f"unsupported PCB revision: {revision}")
    out = output_dir(name, extra, revision)
    out.mkdir(parents=True, exist_ok=True)
    # A failed or interrupted rebuild must not leave an old programmable image
    # alongside the new logs. Source hashes also detect edits during long runs.
    for product in ("top.bit", "top.config", "report.json"):
        (out / product).unlink(missing_ok=True)
    revision_lpf = BOARD["lpf"].with_name(f"v{revision}.lpf")
    sources = [RTL / f for f in RTL_FILES] + [RTL / "rb_isa.vh", BOARD["top"],
                BOARD["lpf"], revision_lpf, ROOT / "fpga" / "build.py"]
    source_hashes = {p.relative_to(ROOT).as_posix(): source_sha256(p) for p in sources}
    (out / "pins.lpf").write_text(BOARD["lpf"].read_text() + "\n" + revision_lpf.read_text())
    files = " ".join((RTL / f).relative_to(ROOT).as_posix() for f in RTL_FILES)
    chparam = "".join(f" -set {k} {v}" for k, v in target["params"].items())
    script = [f"read_verilog -defer -I{RTL.relative_to(ROOT).as_posix()} {files} "
              f"{BOARD['top'].relative_to(ROOT).as_posix()}"]
    if chparam:
        script.append(f"chparam{chparam} top")
    script += [f"synth_ecp5 -top top -json {(out / 'top.json').relative_to(ROOT).as_posix()}",
               "check -assert",
               f"tee -o {(out / 'synth_stat.txt').relative_to(ROOT).as_posix()} stat"]
    (out / "synth.ys").write_text("\n".join(script) + "\n")
    summary: dict = {"target": name, "pcb_revision": revision, "device": target["device"],
                     "package": BOARD["package"], "speed_grade": BOARD["speed"],
                     "clock_mhz": BOARD["freq_mhz"], "params": effective,
                     "seed": seed, "router": router, "sources": source_hashes,
                     "started_utc": datetime.now(timezone.utc).isoformat(), "status": "building"}
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    try:
        for tool, flag in (("yosys", "-V"), ("nextpnr-ecp5", "--version")):
            if synth_only and tool != "yosys":
                continue
            version = subprocess.run([tool, flag], capture_output=True, text=True, check=True)
            summary.setdefault("tools", {})[tool] = (version.stdout + version.stderr).strip()
        print(f"Synthesizing {name}, PCB {revision}...", flush=True)
        summary["synth_seconds"] = round(run(["yosys", "-q", "-l", str(out / "yosys.log"), str(out / "synth.ys")],
                                             out / "yosys.out"), 1)
        stat = (out / "synth_stat.txt").read_text()
        summary["cells"] = {k: int(v) for v, k in re.findall(r"^\s+(\d+)\s+([A-Z][A-Z0-9_]+)$", stat, re.M)}
        summary["status"] = "synthesized"
        if synth_only:
            return summary
        summary["status"] = "routing"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Placing and routing at {BOARD['freq_mhz']} MHz ({router}); log: {out / 'nextpnr.log'}", flush=True)
        summary["pnr_seconds"] = round(run([
            "nextpnr-ecp5", f"--{target['device']}", "--package", BOARD["package"], "--speed", BOARD["speed"],
            "--json", str(out / "top.json"), "--lpf", str(out / "pins.lpf"), "--textcfg", str(out / "top.config"),
            "--freq", str(BOARD["freq_mhz"]), "--report", str(out / "report.json"), "--seed", str(seed),
            "--router", router,
        ], out / "nextpnr.log"), 1)
        report = json.loads((out / "report.json").read_text())
        summary["timing_passed"] = timing_passes(report, BOARD["freq_mhz"])
        summary["fmax_mhz"] = {clk: round(v["achieved"], 2) for clk, v in report.get("fmax", {}).items()}
        summary["utilization"] = {k: v for k, v in report.get("utilization", {}).items() if v.get("used")}
        if not summary["timing_passed"]:
            raise RuntimeError("routed design does not meet the 25 MHz timing constraint; no bitstream produced")
        if source_hashes != {p.relative_to(ROOT).as_posix(): source_sha256(p) for p in sources}:
            raise RuntimeError("sources changed during the build; rebuild before programming")
        run(["ecppack", "--compress", str(out / "top.config"), str(out / "top.bit")], out / "ecppack.log")
        summary["bitstream"] = (out / "top.bit").relative_to(ROOT).as_posix()
        summary["bitstream_sha256"] = sha256(out / "top.bit")
        summary["status"] = "ready"
        return summary
    except (Exception, KeyboardInterrupt) as exc:
        summary.update(status="failed", error=str(exc) or "interrupted")
        (out / "top.bit").unlink(missing_ok=True)
        raise
    finally:
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("target", nargs="?", default="ulx3s-85f")
    p.add_argument("--synth-only", action="store_true")
    p.add_argument("--seed", type=int, default=1, help="placement seed for nextpnr")
    p.add_argument("--router", default="router1", choices=["router1", "router2"])
    p.add_argument("--revision", choices=REVISIONS, default=DEFAULT_REVISION,
                   help="PCB pinout: 3.1.7 for v3.1.6/7 (default); 3.0 for v2.x/v3.0.x")
    p.add_argument("--param", action="append", default=[], metavar="NAME=VALUE",
                   help="override a parameter of fpga/ulx3s/top.v (repeatable)")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    extra = {}
    for item in a.param:
        k, sep, v = item.partition("=")
        if not sep or not k.isidentifier() or not v.isdigit():
            p.error(f"--param expects NAME=VALUE with an integer value, got {item!r}")
        extra[k] = int(v)
    if a.list or a.target not in TARGETS:
        for k, v in TARGETS.items():
            print(f"{k:10s} ECP5-{v['device'].upper()}  {v['params'] or 'default configuration'}")
        return 0 if a.list else 1
    try:
        find_tools(("yosys",) if a.synth_only else ("yosys", "nextpnr-ecp5", "ecppack"))
        summary = build(a.target, a.synth_only, a.seed, a.router, extra, a.revision)
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
