"""cocotb testbench for rb_gpu.

The simulator loads this module; the test body runs in a helper thread (via
cocotb's bridge/resume) so that the ordinary, blocking Device runtime can drive
the simulated host bus. Configuration and the list of scenarios arrive through
environment variables set by tests/rtl/test_rtl.py.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.task import bridge, resume
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "sw", ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from rosenblatt.device import BusBackend, Device, EmuBackend  # noqa: E402
from rosenblatt.emu import Config  # noqa: E402

# counters that must agree exactly between the emulator and the hardware
MATCHED_STATS = ["instructions", "lane_ops", "gmem_instr", "gmem_reqs", "gmem_lanes",
                 "smem_instr", "smem_wavefronts", "divergent_branches", "blocks", "mma_steps"]


class CocotbBackend(BusBackend):
    def __init__(self, dut, cfg: Config):
        self.dut = dut
        self.cfg = cfg

    @resume
    async def bus_write(self, word_addr: int, words: list[int]) -> None:
        dut = self.dut
        for i, w in enumerate(words):
            dut.host_valid.value = 1
            dut.host_we.value = 1
            dut.host_addr.value = word_addr + i
            dut.host_wdata.value = int(w) & 0xFFFF_FFFF
            await RisingEdge(dut.clk)
            while not dut.host_ready.value:
                await RisingEdge(dut.clk)
        dut.host_valid.value = 0
        dut.host_we.value = 0

    @resume
    async def bus_read(self, word_addr: int, n: int) -> list[int]:
        dut = self.dut
        out = []
        for i in range(n):
            dut.host_valid.value = 1
            dut.host_we.value = 0
            dut.host_addr.value = word_addr + i
            await RisingEdge(dut.clk)
            while not dut.host_ready.value:
                await RisingEdge(dut.clk)
            dut.host_valid.value = 0
            while True:
                await RisingEdge(dut.clk)
                if dut.host_rvalid.value:
                    out.append(int(dut.host_rdata.value))
                    break
        return out

    @resume
    async def _wait_busy_low(self) -> None:
        # The launch write was accepted at the edge we last awaited; its effect
        # (busy = 1) is only visible from the next edge on.
        await RisingEdge(self.dut.clk)
        if self.dut.busy.value:
            await FallingEdge(self.dut.busy)
        await RisingEdge(self.dut.clk)

    def wait_idle(self, timeout_s: float = 0) -> None:
        self._wait_busy_low()


async def reset(dut) -> None:
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst.value = 1
    dut.host_valid.value = 0
    dut.host_we.value = 0
    dut.host_addr.value = 0
    dut.host_wdata.value = 0
    await ClockCycles(dut.clk, 4)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


def config_from_env() -> Config:
    return Config(**json.loads(os.environ["RB_CONFIG"]))


def compare(name: str, emu_out, rtl_out) -> list[str]:
    errors = []
    for key, ref in emu_out.outputs.items():
        got = rtl_out.outputs[key]
        if not np.array_equal(ref, got):
            idx = np.argwhere(np.asarray(ref) != np.asarray(got))[:4].tolist()
            errors.append(f"{name}: output '{key}' differs at {idx}")
    for i, (e, r) in enumerate(zip(emu_out.runs, rtl_out.runs)):
        for k in MATCHED_STATS:
            if e.stats.get(k, 0) != r.stats.get(k, 0):
                errors.append(f"{name}[launch {i}]: {k} emu={e.stats.get(k, 0)} rtl={r.stats.get(k, 0)}")
    return errors


def run_scenarios(dut, cfg: Config, names: list[str], seeds: list[int]) -> list[str]:
    from scenarios import SCENARIOS

    rtl = Device(CocotbBackend(dut, cfg))
    hw = rtl.backend.read_config()
    for field in ("num_cores", "num_lanes", "num_warps", "imem_words", "gmem_bytes", "smem_bytes",
                  "stack_depth", "mul_width", "dp4a", "gmem_latency", "gmem_interval", "lsu_depth", "park",
                  "gmem_line_bytes", "scoreboard", "mma"):
        assert getattr(hw, field) == getattr(cfg, field), f"HWCFG {field}: {getattr(hw, field)} != {getattr(cfg, field)}"
    errors: list[str] = []
    for name in names:
        for seed in seeds:
            emu_out = SCENARIOS[name](Device(EmuBackend(cfg)), seed)
            rtl.reset_heap()
            rtl_out = SCENARIOS[name](rtl, seed)
            errs = compare(f"{name}/{seed}", emu_out, rtl_out)
            cyc = sum(r.cycles for r in rtl_out.runs) if rtl_out.runs else 0
            dut._log.info(f"{name:18s} seed {seed}: {'OK' if not errs else 'MISMATCH'}  {cyc} cycles")
            errors += errs
    return errors


@cocotb.test(timeout_time=500, timeout_unit="ms")
async def scenarios(dut):
    await reset(dut)
    cfg = config_from_env()
    names = os.environ["RB_SCENARIOS"].split(",")
    seeds = [int(s) for s in os.environ.get("RB_SEEDS", "0").split(",")]
    errors = await bridge(run_scenarios)(dut, cfg, names, seeds)
    assert not errors, "\n".join(errors)
