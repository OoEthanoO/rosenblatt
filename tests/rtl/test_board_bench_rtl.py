"""The production board benchmark on serial pins, with a partial final tile."""

import pytest

from rosenblatt.emu import Config

import simrun
from test_serial_rtl import UART_SOURCES


@pytest.mark.rtl
def test_board_benchmark():
    out = simrun.ROOT / "build/bench/rtl-board.json"
    out.unlink(missing_ok=True)
    simrun.run(Config(gmem_bytes=64 * 1024), "cocotb_board_bench", {"RB_BENCH_OUT": str(out)},
               toplevel="top", extra_sources=UART_SOURCES + [simrun.ROOT / "fpga/ulx3s/top.v"],
               parameters={"BAUD": 6_250_000}, build_tag="board_bench")
    assert out.is_file()
