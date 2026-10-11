"""Serial protocol faults and full GPU execution through UART pins."""

import pytest

from rosenblatt.emu import Config

import simrun

UART_SOURCES = [simrun.RTL / f"rb_{name}.v" for name in
                ("uart_rx", "uart_tx", "uart_bridge", "serial_gpu")]


@pytest.mark.rtl
@pytest.mark.parametrize("cpb", [4, 9])
def test_uart_bridge(cpb):
    params = {"CLKS_PER_BIT": cpb, "MAX_WORDS": 4,
              "RX_TIMEOUT_CYCLES": cpb * 40, "BUS_TIMEOUT_CYCLES": 40}
    simrun.run(Config(), "cocotb_uart", {"RB_UART_CPB": str(cpb)},
               toplevel="rb_uart_bridge", extra_sources=UART_SOURCES, parameters=params)


@pytest.mark.rtl
@pytest.mark.parametrize("dp4a", [True, False], ids=["packed", "scalar"])
def test_serial_gpu(dp4a):
    cfg = Config(num_cores=1, num_lanes=4, num_warps=2, imem_words=256,
                 gmem_bytes=64 * 1024, smem_bytes=1024, gmem_latency=2,
                 mul_width=32 if dp4a else 16, dp4a=dp4a)
    names = "vecadd,mlp_int8" + (",mlp_trained" if dp4a else "")
    params = {**cfg.verilog_params(), "CLKS_PER_BIT": 4, "MAX_WORDS": 64}
    simrun.run(cfg, "cocotb_serial_gpu", {"RB_UART_CPB": "4", "RB_SCENARIOS": names},
               toplevel="rb_serial_gpu", extra_sources=UART_SOURCES, parameters=params)
