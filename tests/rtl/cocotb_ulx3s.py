"""Board startup, physical UART timing, LED mapping and reset recovery.

The large network test accelerates only the UART divider; the GPU uses the
complete 85F configuration and the same 25 MHz clock in both tests.
"""

import os

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.task import bridge, resume
from cocotb.triggers import ClockCycles, FallingEdge, Timer

from uart_sim import BlockingSerial, SerialPins, ROOT
from cocotb_gpu import compare, config_from_env
from rosenblatt import isa
from rosenblatt.asm import assemble, assemble_file
from rosenblatt.device import Device, EmuBackend
from rosenblatt.uart import Command, UartBackend, request
from scenarios import SCENARIOS


@resume
async def leds(dut):
    return int(dut.led.value)


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
        backend.bus_write(isa.HOST_BOARD_BASE, [13])
        assert (leds(dut) >> 3) & 15 == 13
        assert leds(dut) & 6 == 6  # receive and transmit pulse stretchers
        a = np.arange(9, dtype=np.int32) - 4
        b = np.arange(9, dtype=np.int32) * 17
        pa, pb, pc = dev.upload(a), dev.upload(b), dev.alloc(a.nbytes)
        dev.load(assemble_file(str(ROOT / "kernels/vecadd.s")))
        run = dev.launch(grid=2, block=8, args=[pa, pb, pc, len(a)])
        assert np.array_equal(dev.download(pc, len(a)), a + b)
        assert run.cycles > 0 and all(core["blocks"] == 1 for core in run.per_core)
        assert leds(dut) & 1 == 0
        if full_network:
            for name in ("reduce_sum", "mlp_trained", "mlp_tiled"):
                dev.reset_heap()
                expected = SCENARIOS[name](Device(EmuBackend(dev.cfg)), 0)
                actual = SCENARIOS[name](dev, 0)
                assert not (errors := compare(name, expected, actual)), "\n".join(errors)
                dut._log.info("ULX3S %s: outputs and counters match", name)
        # Leave both cores running. The physical reset button must recover the
        # whole device, even with an unfinished packet in the serial bridge.
        dev.load(assemble("spin: bra spin"))
        backend.bus_write(isa.HOST_CSR_BASE + isa.CSR.GRID, [2, 8, 0])
        backend.bus_write(isa.HOST_CSR_BASE + isa.CSR.CTRL, [1])
        assert leds(dut) & 1 == 1


def check_recovered(dut, pins):
    with connect(pins) as backend:
        assert leds(dut) & 0x79 == 0  # busy and the displayed digit reset
        dev = Device(backend)
        dev.load(assemble("ret"))
        assert dev.launch(grid=2, block=8).errors == 0


@cocotb.test(timeout_time=2, timeout_unit="sec")
async def board(dut):
    dut.btn.value = 0
    dut.ftdi_txd.value = 1
    cocotb.start_soon(Clock(dut.clk_25mhz, 40, unit="ns").start())
    await ClockCycles(dut.clk_25mhz, 20)
    assert dut.rst.value == 1 and dut.ftdi_rxd.value == 1
    assert dut.wifi_gpio0.value == 1
    await ClockCycles(dut.clk_25mhz, 65540)
    await FallingEdge(dut.clk_25mhz)
    assert dut.rst.value == 0
    baud = int(os.environ["RB_BOARD_BAUD"])
    pins = SerialPins(dut, clk=dut.clk_25mhz, rx=dut.ftdi_txd,
                      tx=dut.ftdi_rxd, period_ns=40, baudrate=baud)
    await bridge(run_checks)(dut, pins, baud != 115200)
    await pins.send(request(Command.WRITE, 42, 0, 1, b"\xff" * 4)[:-3])
    # Assert off the clock edge to exercise the button synchronizer.
    await Timer(7, unit="ns")
    dut.btn.value = 2
    await ClockCycles(dut.clk_25mhz, 10)
    assert dut.rst.value == 1
    assert int(dut.led.value) & 0x79 == 0
    dut.btn.value = 0
    await ClockCycles(dut.clk_25mhz, 100)
    assert dut.rst.value == 1
    # Contact bounce starts the hold interval again.
    dut.btn.value = 2
    await ClockCycles(dut.clk_25mhz, 10)
    dut.btn.value = 0
    await ClockCycles(dut.clk_25mhz, 65540)
    await FallingEdge(dut.clk_25mhz)
    assert dut.rst.value == 0
    await bridge(check_recovered)(dut, pins)
