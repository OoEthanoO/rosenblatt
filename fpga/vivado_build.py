"""Build the GPU for an AMD (Xilinx) FPGA board with Vivado.

    python fpga/vivado_build.py basys3               # bitstream + resource/timing reports
    python fpga/vivado_build.py basys3 --synth-only  # just count resources
    python fpga/vivado_build.py basys3 --param SCOREBOARD=1   # a variant
    python fpga/vivado_build.py --list

Outputs go to build/fpga/<target>/: top.bit, Vivado's log and reports, and
summary.json with the fields fpga/build.py writes for the ULX3S, so
`python fpga/bringup.py --board basys3` checks and loads the image the same
way. Vivado (the free Standard edition) comes from AMD; this script finds it
on PATH, in the default install folders, or at $VIVADO (the vivado executable
or the folder that contains bin/).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fpga import build as ecp5_build  # noqa: E402  (the RTL file list and checksums are shared)

RTL = ROOT / "rtl"
TARGETS = {
    # Digilent Basys 3: Artix-7 XC7A35T with a 100 MHz oscillator; one core of the default design
    "basys3": {"part": "xc7a35tcpg236-1", "top": ROOT / "fpga" / "basys3" / "top.v",
               "xdc": ROOT / "fpga" / "basys3" / "basys3.xdc", "params": {}},
}
UTILIZATION = {"Slice LUTs": "luts", "LUT as Memory": "luts_as_memory", "Slice Registers": "registers",
               "Block RAM Tile": "bram_tiles", "DSPs": "dsps"}


def output_dir(name: str, extra: dict[str, int] | None = None) -> Path:
    return ROOT / "build" / "fpga" / (name + "".join(f"+{k}={v}" for k, v in sorted((extra or {}).items())))


def _version(path: Path) -> tuple[int, int]:
    m = re.search(r"(20\d\d)\.(\d+)", path.as_posix())
    return (int(m[1]), int(m[2])) if m else (0, 0)


def find_vivado() -> str:
    """$VIVADO, then PATH, then the newest version in the default install folders."""
    env = os.environ.get("VIVADO")
    if env:
        p = Path(env)
        for c in ([p] if p.is_file() else [p / "bin" / "vivado.bat", p / "bin" / "vivado"]):
            if c.is_file():
                return str(c)
        raise RuntimeError(f"VIVADO={env} is neither the vivado executable nor its install folder")
    found = shutil.which("vivado")
    if found:
        return found
    patterns = (["C:/AMDDesignTools/*/Vivado/bin/vivado.bat", "C:/Xilinx/*/Vivado/bin/vivado.bat",
                 "C:/Xilinx/Vivado/*/bin/vivado.bat"] if os.name == "nt" else
                ["/tools/Xilinx/*/Vivado/bin/vivado", "/tools/Xilinx/Vivado/*/bin/vivado",
                 "/opt/Xilinx/*/Vivado/bin/vivado", "/opt/Xilinx/Vivado/*/bin/vivado"])
    candidates = [Path(p) for pattern in patterns for p in glob.glob(pattern)]
    if candidates:
        return str(max(candidates, key=_version))
    raise RuntimeError("vivado not found: install Vivado (see docs/fpga.md) or set VIVADO to its folder")


def run_vivado(tcl: Path, log: Path) -> float:
    """Run a Tcl script in batch mode. Vivado's scratch files stay beside the log."""
    t = time.time()
    with open(log.with_suffix(".out"), "w", encoding="utf-8") as f:
        p = subprocess.run([find_vivado(), "-mode", "batch", "-nojournal", "-log", str(log), "-source", str(tcl)],
                           stdout=f, stderr=subprocess.STDOUT, cwd=log.parent)
    if p.returncode != 0:
        tail = log.with_suffix(".out").read_text(errors="replace").splitlines()[-25:]
        raise RuntimeError(f"vivado exited with {p.returncode} (see {log}):\n" + "\n".join(tail))
    return time.time() - t


def _tcl_path(p: Path) -> str:
    return "{" + p.as_posix() + "}"


def build_script(target: dict, params: dict[str, int], out: Path, synth_only: bool) -> str:
    sources = " ".join(_tcl_path(RTL / f) for f in ecp5_build.RTL_FILES) + " " + _tcl_path(target["top"])
    generics = "".join(f" -generic {k}={v}" for k, v in sorted(params.items()))
    lines = [
        f"read_verilog [list {sources}]",
        f"read_xdc {_tcl_path(target['xdc'])}",
        f"synth_design -top top -part {target['part']} -include_dirs {_tcl_path(RTL)} "
        f"-verilog_define SYNTHESIS=1{generics}",
        f"report_utilization -file {_tcl_path(out / 'utilization_synth.txt')}",
    ]
    if not synth_only:
        lines += [
            "opt_design", "place_design", "phys_opt_design", "route_design",
            f"report_utilization -file {_tcl_path(out / 'utilization.txt')}",
            f"report_timing_summary -file {_tcl_path(out / 'timing.txt')}",
            # one line per clock: name, period (ns), worst setup and hold slack (ns) or none
            f"set f [open {_tcl_path(out / 'clocks.txt')} w]",
            "foreach c [get_clocks] {",
            "    set setup [get_timing_paths -quiet -delay_type max -group $c -max_paths 1]",
            "    set hold [get_timing_paths -quiet -delay_type min -group $c -max_paths 1]",
            '    set ws [expr {[llength $setup] ? [get_property SLACK $setup] : "none"}]',
            '    set wh [expr {[llength $hold] ? [get_property SLACK $hold] : "none"}]',
            '    puts $f "[get_property NAME $c] [get_property PERIOD $c] $ws $wh"',
            "}",
            "close $f",
            # no bitstream unless every setup and hold check passes
            "set setup [get_timing_paths -quiet -delay_type max -max_paths 1]",
            "set hold [get_timing_paths -quiet -delay_type min -max_paths 1]",
            "if {[llength $setup] && [get_property SLACK $setup] >= 0 &&",
            "    [llength $hold] && [get_property SLACK $hold] >= 0} {",
            f"    write_bitstream -force {_tcl_path(out / 'top.bit')}",
            "}",
        ]
    return "\n".join(lines) + "\n"


def parse_utilization(text: str) -> dict[str, dict[str, float]]:
    """The first occurrence of each row of interest in report_utilization's tables."""
    found: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        name = cells[0].rstrip("*").strip() if cells else ""
        if name in UTILIZATION and UTILIZATION[name] not in found and len(cells) >= 4:
            used, available = float(cells[1]), float(cells[-2])
            found[UTILIZATION[name]] = {"used": int(used) if used.is_integer() else used,
                                        "available": int(available)}
    return found


def parse_clocks(text: str) -> dict[str, dict[str, float | None]]:
    clocks = {}
    for line in text.splitlines():
        if line.strip():
            name, period, setup, hold = line.split()
            clocks[name] = {"period_ns": float(period), "setup_slack_ns": None if setup == "none" else float(setup),
                            "hold_slack_ns": None if hold == "none" else float(hold)}
    return clocks


def timing_passes(clocks: dict) -> bool:
    timed = [c for c in clocks.values() if c["setup_slack_ns"] is not None]
    return bool(timed) and all(c["setup_slack_ns"] >= 0 and (c["hold_slack_ns"] or 0) >= 0 for c in timed)


def build(name: str, synth_only: bool = False, extra: dict[str, int] | None = None) -> dict:
    target = dict(TARGETS[name])
    defaults = {k: int(v.replace("_", "")) for k, v in re.findall(
        r"parameter\s+(\w+)\s*=\s*([\d_]+)", target["top"].read_text())}
    params = {**target["params"], **(extra or {})}
    unknown = set(params) - defaults.keys()
    if unknown:
        raise ValueError(f"unknown top-level parameters: {', '.join(sorted(unknown))}")
    effective = {**defaults, **params}
    divide = 1e9 / effective["CLK_HZ"]   # the MMCM's output divider from its 1 GHz VCO
    if not 2 <= divide <= 128 or (divide * 8) % 1:
        raise ValueError("CLK_HZ must be 1 GHz divided by 2..128 in steps of 1/8, e.g. 25000000")
    if effective["BAUD"] <= 0 or effective["BAUD"] > effective["CLK_HZ"] // 4:
        raise ValueError("BAUD must be positive and allow at least four clocks per bit")
    out = output_dir(name, extra)
    out.mkdir(parents=True, exist_ok=True)
    # A failed or interrupted rebuild must not leave an old programmable image
    # alongside the new reports. Source checksums also detect edits during a run.
    for product in ("top.bit", "clocks.txt", "utilization.txt", "timing.txt"):
        (out / product).unlink(missing_ok=True)
    sources = [RTL / f for f in ecp5_build.RTL_FILES] + [RTL / "rb_isa.vh", target["top"], target["xdc"],
                                                        ROOT / "fpga" / "vivado_build.py"]
    source_hashes = {p.relative_to(ROOT).as_posix(): ecp5_build.source_sha256(p) for p in sources}
    (out / "build.tcl").write_text(build_script(target, params, out, synth_only))
    summary: dict = {"target": name, "device": target["part"], "clock_mhz": effective["CLK_HZ"] / 1e6,
                     "params": effective, "sources": source_hashes,
                     "started_utc": datetime.now(timezone.utc).isoformat(), "status": "building"}
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    try:
        version = subprocess.run([find_vivado(), "-version"], capture_output=True, text=True)
        summary["tools"] = {"vivado": (version.stdout + version.stderr).strip().splitlines()[0]}
        print(f"Running Vivado for {name} ({'synthesis' if synth_only else 'synthesis, place and route'}); "
              f"log: {out / 'vivado.log'}", flush=True)
        summary["vivado_seconds"] = round(run_vivado(out / "build.tcl", out / "vivado.log"), 1)
        report = out / ("utilization_synth.txt" if synth_only else "utilization.txt")
        summary["utilization"] = parse_utilization(report.read_text())
        if synth_only:
            summary["status"] = "synthesized"
            return summary
        clocks = parse_clocks((out / "clocks.txt").read_text())
        summary["timing"] = clocks
        summary["timing_passed"] = timing_passes(clocks)
        summary["fmax_mhz"] = {n: round(1000 / (c["period_ns"] - c["setup_slack_ns"]), 2)
                               for n, c in clocks.items() if c["setup_slack_ns"] is not None}
        if not summary["timing_passed"]:
            raise RuntimeError("routed design does not meet timing; no bitstream produced")
        if not (out / "top.bit").is_file():
            raise RuntimeError("vivado wrote no bitstream; see vivado.log")
        if source_hashes != {p.relative_to(ROOT).as_posix(): ecp5_build.source_sha256(p) for p in sources}:
            raise RuntimeError("sources changed during the build; rebuild before programming")
        summary["bitstream"] = (out / "top.bit").relative_to(ROOT).as_posix()
        summary["bitstream_sha256"] = ecp5_build.sha256(out / "top.bit")
        summary["status"] = "ready"
        return summary
    except (Exception, KeyboardInterrupt) as exc:
        summary.update(status="failed", error=str(exc) or "interrupted")
        (out / "top.bit").unlink(missing_ok=True)
        raise
    finally:
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")


def program(bitstream: Path, part: str, serial_number: str | None = None) -> None:
    """Load the FPGA's configuration memory over JTAG with Vivado's hardware
    manager (volatile: the board's flash keeps its own image)."""
    # A Digilent board's USB serial number ends in a letter for each FTDI
    # channel: B for the serial port, A for JTAG. Match the rest.
    serial = serial_number[:-1] if serial_number and serial_number[-1] in "AB" else serial_number
    pick = f"lsearch -inline -glob [get_hw_targets] {{*{serial}*}}" if serial else "lindex [get_hw_targets] 0"
    script = bitstream.with_name("program.tcl")
    script.write_text("\n".join([
        "open_hw_manager",
        "connect_hw_server",
        f"set target [{pick}]",
        'if {$target eq ""} { error "no matching JTAG target; is the board connected and switched on?" }',
        "open_hw_target $target",
        f"set device [lindex [get_hw_devices {part}*] 0]",
        "current_hw_device $device",
        f"set_property PROGRAM.FILE {_tcl_path(bitstream)} $device",
        "program_hw_devices $device",
        "close_hw_target",
        "disconnect_hw_server",
    ]) + "\n")
    run_vivado(script, bitstream.with_name("program.log"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("target", nargs="?", default="basys3")
    p.add_argument("--synth-only", action="store_true")
    p.add_argument("--param", action="append", default=[], metavar="NAME=VALUE",
                   help="override a parameter of the board's top.v (repeatable)")
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
            print(f"{k:10s} {v['part']}  {v['params'] or 'default configuration'}")
        return 0 if a.list else 1
    try:
        summary = build(a.target, a.synth_only, extra)
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
