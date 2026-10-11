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
    # odd corners: one core, 4 lanes, 16-bit multiplier, no dp4a, 1-cycle memory
    "small": Config(num_cores=1, num_lanes=4, num_warps=2, mul_width=16, dp4a=False,
                    gmem_latency=1, smem_bytes=1024, stack_depth=8),
    # slow, narrow memory and more cores contending for it
    "busy_mem": Config(num_cores=3, num_lanes=4, num_warps=4, gmem_latency=13, gmem_interval=2),
}


@pytest.mark.rtl
@pytest.mark.parametrize("cfg_name", list(CONFIGS))
def test_scenarios(cfg_name):
    cfg = CONFIGS[cfg_name]
    seeds = "0,1,2" if cfg_name == "default" else "0"
    simrun.run(cfg, "cocotb_gpu", {"RB_SCENARIOS": ",".join(SCENARIOS), "RB_SEEDS": seeds})
