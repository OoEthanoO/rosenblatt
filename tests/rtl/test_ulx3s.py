"""Exercise the actual 85F board wrapper, including its clock and reset."""

import pytest

from rosenblatt.emu import Config

import simrun
from test_serial_rtl import UART_SOURCES


@pytest.mark.rtl
@pytest.mark.parametrize("baud", [115200, 6250000], ids=["physical_baud", "full_network"])
def test_ulx3s(baud):
    simrun.run(Config(gmem_bytes=64 * 1024), "cocotb_ulx3s", {"RB_BOARD_BAUD": str(baud)},
               toplevel="top", extra_sources=UART_SOURCES + [simrun.ROOT / "fpga/ulx3s/top.v"],
               parameters={"BAUD": baud})
