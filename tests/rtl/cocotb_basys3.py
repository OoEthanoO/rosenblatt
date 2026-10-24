"""Basys 3 wrapper: clock, startup, physical UART timing, LEDs, display and reset recovery.

The large network test accelerates only the UART divider; the GPU runs on the
25 MHz clock made from the simulated 100 MHz oscillator in both tests.
"""

import os

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.task import bridge, resume
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge, Timer
from cocotb.utils import get_sim_time

from uart_sim import BlockingSerial, SerialPins, ROOT
from cocotb_gpu import compare, config_from_env
from rosenblatt import isa
from rosenblatt.asm import assemble, assemble_file
from rosenblatt.device import Device, EmuBackend
from rosenblatt.uart import Command, UartBackend, request
from scenarios import SCENARIOS

DASH = 0x7F & ~0x40          # segment outputs are active low; a dash lights only g
SEVEN = 0x7F & ~0x07         # 7: segments a, b, c


@resume
async def board_outputs(dut):
    return int(dut.led.value), int(dut.seg.value)


def connect(pins):
    backend = UartBackend(stream=BlockingSerial(pins), timeout=180, kernel_timeout=180)
    assert backend.cfg == config_from_env()
    return backend


def run_checks(dut, pins, full_network):
    with connect(pins) as backend:
        dev = Device(backend)
        # Catch truncated address bits and preserve bytes on both sides of an
        # unaligned write near the end of the physical 64 KiB memory.
        base = dev.cfg.gmem_bytes - 32
        dev.write(base, b"Z" * 32)
        dev.write(base + 3, bytes(range(23)))
        assert dev.read(base, 32) == b"Z" * 3 + bytes(range(23)) + b"Z" * 6
        backend.bus_write(isa.HOST_BOARD_BASE, [7])
        led, seg = board_outputs(dut)
        assert (led >> 3) & 15 == 7
        assert led & 6 == 6           # receive and transmit pulse stretchers
        assert seg == DASH            # no kernel has run yet
        a = np.arange(9, dtype=np.int32) - 4
        b = np.arange(9, dtype=np.int32) * 17
        pa, pb, pc = dev.upload(a), dev.upload(b), dev.alloc(a.nbytes)
        dev.load(assemble_file(str(ROOT / "kernels/vecadd.s")))
        run = dev.launch(grid=2, block=8, args=[pa, pb, pc, len(a)])
        assert np.array_equal(dev.download(pc, len(a)), a + b)
        assert run.cycles > 0 and [core["blocks"] for core in run.per_core] == [2]
        led, seg = board_outputs(dut)
        assert led & 1 == 0 and seg == SEVEN
        if full_network:
            for name in ("reduce_sum", "mlp_trained", "mlp_tiled"):
                dev.reset_heap()
                expected = SCENARIOS[name](Device(EmuBackend(dev.cfg)), 0)
                actual = SCENARIOS[name](dev, 0)
                assert not (errors := compare(name, expected, actual)), "\n".join(errors)
                dut._log.info("Basys 3 %s: outputs and counters match", name)
        # Leave the core running. The reset button must recover the whole
        # device, even with an unfinished packet in the serial bridge.
        dev.load(assemble("spin: bra spin"))
        backend.bus_write(isa.HOST_CSR_BASE + isa.CSR.GRID, [1, 8, 0])
        backend.bus_write(isa.HOST_CSR_BASE + isa.CSR.CTRL, [1])
        assert board_outputs(dut)[0] & 1 == 1


def check_recovered(dut, pins):
    with connect(pins) as backend:
        led, seg = board_outputs(dut)
        assert led & 0x79 == 0 and seg == DASH   # busy, the value and the display reset
        dev = Device(backend)
        dev.load(assemble("ret"))
        assert dev.launch(grid=2, block=8).errors == 0


@cocotb.test(timeout_time=2, timeout_unit="sec")
async def board(dut):
    dut.btnC.value = 0
    dut.RsRx.value = 1
    cocotb.start_soon(Clock(dut.clk_100mhz, 10, unit="ns").start())
    await ClockCycles(dut.clk_100mhz, 100)
    await RisingEdge(dut.clk)
    t = get_sim_time("ns")
    await RisingEdge(dut.clk)
    assert get_sim_time("ns") - t == 40, "the GPU clock is not 25 MHz"
    assert dut.rst.value == 1 and dut.RsTx.value == 1
    assert int(dut.seg.value) == DASH and int(dut.an.value) == 0b1110 and dut.dp.value == 1
    await ClockCycles(dut.clk, 65540)
    await FallingEdge(dut.clk)
    assert dut.rst.value == 0
    baud = int(os.environ["RB_BOARD_BAUD"])
    pins = SerialPins(dut, clk=dut.clk, rx=dut.RsRx, tx=dut.RsTx, period_ns=40, baudrate=baud)
    await bridge(run_checks)(dut, pins, baud != 115200)
    await pins.send(request(Command.WRITE, 42, 0, 1, b"\xff" * 4)[:-3])
    # Press off the clock edge to exercise the button synchronizer.
    await Timer(7, unit="ns")
    dut.btnC.value = 1
    await ClockCycles(dut.clk, 10)
    assert dut.rst.value == 1
    assert int(dut.led.value) & 0x79 == 0 and int(dut.seg.value) == DASH
    dut.btnC.value = 0
    await ClockCycles(dut.clk, 100)
    assert dut.rst.value == 1
    # Contact bounce starts the hold interval again.
    dut.btnC.value = 1
    await ClockCycles(dut.clk, 10)
    dut.btnC.value = 0
    await ClockCycles(dut.clk, 65540)
    await FallingEdge(dut.clk)
    assert dut.rst.value == 0
    await bridge(check_recovered)(dut, pins)
