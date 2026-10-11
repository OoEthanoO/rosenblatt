"""Build and run the RTL in a simulator from Python (pytest, scripts, benchmarks)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

from rosenblatt.emu import Config

ROOT = Path(__file__).resolve().parents[2]
RTL = ROOT / "rtl"
SOURCES = ["rb_gpu.v", "rb_core.v", "rb_lsu.v", "rb_smem.v", "rb_alu.v", "rb_regfile.v", "rb_gmem.v"]
OSS_CAD = Path(os.environ.get("OSS_CAD_SUITE", Path.home() / "tools" / "oss-cad-suite"))


def tool_env() -> None:
    """Put the OSS CAD Suite tools on PATH if they are not already there."""
    if shutil.which("iverilog") is None and (OSS_CAD / "bin").is_dir():
        os.environ["PATH"] = os.pathsep.join([str(OSS_CAD / "bin"), str(OSS_CAD / "lib"),
                                              os.environ.get("PATH", "")])
    os.environ.setdefault("VERILATOR_ROOT", str(OSS_CAD / "share" / "verilator"))


def run(cfg: Config, test_module: str, env: dict[str, str], sim: str = "icarus",
        waves: bool = False, test_dir: Path | None = None, toplevel: str = "rb_gpu",
        extra_sources: list[Path] | None = None) -> None:
    from cocotb_tools.runner import get_runner

    tool_env()
    params = cfg.verilog_params()
    tag = hashlib.sha1(json.dumps([params, toplevel], sort_keys=True).encode()).hexdigest()[:10]
    build_dir = ROOT / "build" / "sim" / f"{sim}_{toplevel}_{tag}"
    runner = get_runner(sim)
    sources = [RTL / s for s in SOURCES] + list(extra_sources or [])
    runner.build(sources=sources, includes=[RTL], hdl_toplevel=toplevel, parameters=params,
                 build_dir=build_dir, timescale=("1ns", "1ps"), waves=waves)
    full_env = {"RB_CONFIG": json.dumps(asdict(cfg))}
    full_env.update(env)
    runner.test(hdl_toplevel=toplevel, test_module=test_module,
                test_dir=test_dir or Path(__file__).parent, build_dir=build_dir,
                extra_env=full_env, waves=waves)
