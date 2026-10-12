"""Serial pins and a blocking pyserial-shaped adapter for cocotb tests."""

import os
import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.queue import Queue
from cocotb.task import resume
from cocotb.triggers import ClockCycles, FallingEdge, Timer, with_timeout

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "sw", ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

PERIOD_NS = 10


async def reset(dut, *, start_clock=True):
    if start_clock:
        cocotb.start_soon(Clock(dut.clk, PERIOD_NS, unit="ns").start())
    dut.uart_rx.value = 1
    dut.rst.value = 1
    await ClockCycles(dut.clk, 5)
    await FallingEdge(dut.clk)
    dut.rst.value = 0
    await ClockCycles(dut.clk, 5)


class SerialPins:
    def __init__(self, dut, *, clk=None, rx=None, tx=None, period_ns=PERIOD_NS, baudrate=None):
        self.dut = dut
        self.clk = dut.clk if clk is None else clk
        self.rx = dut.uart_rx if rx is None else rx
        self.tx = dut.uart_tx if tx is None else tx
        self.bit_ns = 1e9 / baudrate if baudrate else int(os.environ["RB_UART_CPB"]) * period_ns
        self.send_bit_ns = self.bit_ns
        self.received = Queue()
        cocotb.start_soon(self._receive())

    async def _receive(self):
        while True:
            await FallingEdge(self.tx)
            await Timer(self.bit_ns * 1.5, unit="ns", round_mode="round")
            value = 0
            for bit in range(8):
                value |= int(self.tx.value) << bit
                await Timer(self.bit_ns, unit="ns", round_mode="round")
            assert int(self.tx.value) == 1, "invalid UART TX stop bit"
            self.received.put_nowait(value)

    async def send(self, data, *, stop_bit=1):
        await FallingEdge(self.clk)
        for value in data:
            for bit in [0] + [(value >> b) & 1 for b in range(8)] + [stop_bit]:
                self.rx.value = bit
                await Timer(self.send_bit_ns, unit="ns", round_mode="round")
        self.rx.value = 1

    async def read(self, n):
        data = bytearray()
        for _ in range(n):
            data.append(await with_timeout(self.received.get(), 2, "ms"))
        return bytes(data)


class BlockingSerial:
    """UartBackend runs in a bridge thread; all I/O crosses the serial pins."""

    def __init__(self, pins):
        self.pins = pins
        self.timeout = self.write_timeout = 120.0

    @resume
    async def write(self, data):
        await self.pins.send(data)
        return len(data)

    @resume
    async def read(self, n):
        return await self.pins.read(n)
