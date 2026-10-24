"""pytest entry point for the RTL: every scenario, several hardware configurations.

    pytest tests/rtl -m rtl            (or simply `pytest`)
"""

from __future__ import annotations

import pytest

from rosenblatt.emu import Config
from scenarios import SCENARIOS

import simrun

CONFIGS = {
    # the default build: 2 cores x 8 lanes x 4 warps
    "default": Config(),
    # odd corners: one core, 4 lanes, 16-bit multiplier, no dp4a, 1-cycle memory,
    # one memory instruction in flight, register scoreboard
    "small": Config(num_cores=1, num_lanes=4, num_warps=2, mul_width=16, dp4a=False,
                    gmem_latency=1, smem_bytes=1024, stack_depth=8, lsu_depth=1, scoreboard=True),
    # slow, narrow memory, more cores contending for it, parked warps, 8-byte lines,
    # a matrix unit with 4 x 4 tiles and four operations per warp in it at once
    "busy_mem": Config(num_cores=3, num_lanes=4, num_warps=4, gmem_latency=13, gmem_interval=2,
                       lsu_depth=2, park=True, gmem_line_bytes=8, mma=True, mma_slots=4),
    # the default GPU with 32-byte memory lines (one request can serve a whole warp),
    # a register scoreboard (a load stalls only the instructions that need it)
    # and the matrix unit
    "wide_lines": Config(gmem_line_bytes=32, scoreboard=True, mma=True),
    # the same with two operations per warp in the matrix unit at once
    "two_slots": Config(gmem_line_bytes=32, scoreboard=True, mma=True, mma_slots=2),
    # one core with eight warps, a slow memory, an eight-instruction load/store
    # queue, the scoreboard and the matrix unit
    "eight_warps": Config(num_cores=1, num_warps=8, gmem_latency=32, lsu_depth=8, scoreboard=True,
                          mma=True, mma_slots=2),
}


@pytest.mark.rtl
@pytest.mark.parametrize("cfg_name", list(CONFIGS))
def test_scenarios(cfg_name):
    cfg = CONFIGS[cfg_name]
    seeds = "0,1,2" if cfg_name == "default" else "0"
    simrun.run(cfg, "cocotb_gpu", {"RB_SCENARIOS": ",".join(SCENARIOS), "RB_SEEDS": seeds})
