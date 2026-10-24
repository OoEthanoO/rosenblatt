"""Exercise the Basys 3 board wrapper, including its clock and reset."""

import pytest

from rosenblatt.emu import Config

import simrun
from test_serial_rtl import UART_SOURCES


@pytest.mark.rtl
@pytest.mark.parametrize("baud", [115200, 6250000], ids=["physical_baud", "full_network"])
def test_basys3(baud):
    simrun.run(Config(num_cores=1, gmem_bytes=64 * 1024), "cocotb_basys3", {"RB_BOARD_BAUD": str(baud)},
               toplevel="top", extra_sources=UART_SOURCES + [simrun.ROOT / "fpga/basys3/top.v"],
               parameters={"BAUD": baud}, build_tag="basys3")
