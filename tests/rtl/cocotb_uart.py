"""Exercise the bridge with delayed/stalled bus handshakes and real UART bits."""

import binascii
import os
import struct

import cocotb
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge

from uart_sim import SerialPins, reset
from rosenblatt.uart import Command, Status, request


class MemoryBus:
    def __init__(self, dut):
        self.dut = dut
        self.memory = {}
        self.transactions = []
        self.stall = False
        self.drop_reads = False
        self.latency = 3
        cocotb.start_soon(self.run())

    async def run(self):
        dut = self.dut
        cycle = 0
        pending = None
        held = None
        while True:
            await FallingEdge(dut.clk)
            fire = False
            if dut.rst.value:
                dut.host_ready.value = 0
                dut.host_rvalid.value = 0
                pending = held = None
            else:
                ready = not self.stall and cycle % 5 not in (1, 2)
                valid = int(dut.host_valid.value)
                write = int(dut.host_we.value)
                addr = int(dut.host_addr.value)
                data = int(dut.host_wdata.value) if valid and write else 0
                signals = (write, addr, data)
                if held is not None and not self.stall:
                    assert valid and signals == held, "request changed while bus was stalled"
                held = signals if valid and not ready else None
                fire = valid and ready
                rvalid, rdata = False, 0
                if pending is not None and cycle >= pending[0]:
                    rvalid, rdata = True, pending[1]
                    pending = None
                if fire and not write and self.latency == 0 and not self.drop_reads:
                    rvalid, rdata = True, self.memory.get(addr, 0)
                dut.host_ready.value = int(ready)
                dut.host_rvalid.value = int(rvalid)
                dut.host_rdata.value = rdata
            await RisingEdge(dut.clk)
            if fire:
                self.transactions.append(signals)
                if write:
                    self.memory[addr] = data
                elif self.latency and not self.drop_reads:
                    assert pending is None
                    pending = (cycle + self.latency, self.memory.get(addr, 0))
            cycle += 1


async def exchange(pins, packet, *, expected=Status.OK):
    await pins.send(packet)
    header = await pins.read(5)
    assert header[:2] == b"RB"
    assert header[2] == packet[3], "response sequence mismatch"
    assert header[3] == expected, (header[3], expected)
    tail = await pins.read(header[4] * 4 + 2)
    assert binascii.crc_hqx(header[2:] + tail[:-2], 0xFFFF) == int.from_bytes(tail[-2:], "little")
    return list(struct.unpack(f"<{header[4]}I", tail[:-2]))


async def setup(dut):
    dut.host_ready.value = 0
    dut.host_rvalid.value = 0
    dut.host_rdata.value = 0
    await reset(dut)
    return SerialPins(dut), MemoryBus(dut)


@cocotb.test(timeout_time=20, timeout_unit="ms")
async def bursts_and_validation(dut):
    pins, bus = await setup(dut)
    assert await exchange(pins, request(Command.HELLO, 0, 0, 0)) == [0x0401]
    # All 256 byte values, embedded sync bytes, and both immediate/delayed reads.
    words = list(struct.unpack("<64I", bytes(range(256))))
    for offset in range(0, len(words), 4):
        group = words[offset:offset + 4]
        assert await exchange(pins, request(Command.WRITE, offset, 100 + offset, 4,
                                           struct.pack("<4I", *group))) == []
        bus.latency = 0 if offset % 8 else 3
        assert await exchange(pins, request(Command.READ, offset + 1, 100 + offset, 4)) == group
    assert [bus.memory[100 + i] for i in range(64)] == words
    # A one-word transfer at the last valid address is legal; wraparound is not.
    await exchange(pins, request(Command.WRITE, 70, 0xFFFFFF, 1, b"\xFF" * 4))
    assert await exchange(pins, request(Command.READ, 71, 0xFFFFFF, 1)) == [0xFFFFFFFF]
    before = len(bus.transactions)
    corrupt = bytearray(request(Command.WRITE, 72, 5, 1, b"abcd"))
    corrupt[8] ^= 1
    await exchange(pins, bytes(corrupt), expected=Status.BAD_CRC)
    for packet, status in [
        (request(0x7F, 73, 0, 1), Status.BAD_COMMAND),
        (request(Command.READ, 74, 0, 0), Status.BAD_LENGTH),
        (request(Command.WRITE, 75, 0, 5, bytes(20)), Status.BAD_LENGTH),
        (request(Command.READ, 76, 0xFFFFFF, 2), Status.BAD_ADDRESS),
        (request(Command.HELLO, 77, 0, 1), Status.BAD_LENGTH),
        (request(Command.HELLO, 78, 1, 0), Status.BAD_ADDRESS),
    ]:
        assert await exchange(pins, packet, expected=status) == []
    assert len(bus.transactions) == before, "invalid request reached the bus"
    assert await exchange(pins, request(Command.HELLO, 255, 0, 0)) == [0x0401]


@cocotb.test(timeout_time=20, timeout_unit="ms")
async def truncated_frames_and_bad_stop_bits(dut):
    pins, bus = await setup(dut)
    cpb = int(os.environ["RB_UART_CPB"])
    packet = request(Command.WRITE, 1, 0, 4, b"RB" * 8)
    await pins.send(packet[:-3])
    await ClockCycles(dut.clk, cpb * 50)
    assert not bus.transactions
    assert await exchange(pins, request(Command.HELLO, 2, 0, 0)) == [0x0401]
    await pins.send(packet[:-2])
    await pins.send(b"\x00", stop_bit=0)
    await ClockCycles(dut.clk, cpb * 15)
    assert not bus.transactions
    assert await exchange(pins, request(Command.HELLO, 3, 0, 0)) == [0x0401]
    # A short low glitch must not become a byte. Overlapping sync prefixes recover.
    await FallingEdge(dut.clk)
    dut.uart_rx.value = 0
    await ClockCycles(dut.clk, 1)
    dut.uart_rx.value = 1
    await ClockCycles(dut.clk, cpb * 15)
    await pins.send(b"noiseR")
    assert await exchange(pins, request(Command.HELLO, 4, 0, 0)) == [0x0401]
    if cpb >= 9:
        for scale in (0.99, 1.01):
            pins.send_bit_ns = pins.bit_ns * scale
            assert await exchange(pins, request(Command.HELLO, 5, 0, 0)) == [0x0401]


@cocotb.test(timeout_time=20, timeout_unit="ms")
async def bus_timeouts_require_reset(dut):
    pins, bus = await setup(dut)
    for mode in ("stall", "drop_reads"):
        setattr(bus, mode, True)
        await exchange(pins, request(Command.READ, 1, 0, 1), expected=Status.BUS_TIMEOUT)
        setattr(bus, mode, False)
        before = len(bus.transactions)
        # A late bus response cannot be mistaken for a new request's response.
        await exchange(pins, request(Command.HELLO, 2, 0, 0), expected=Status.BUS_TIMEOUT)
        await exchange(pins, request(Command.READ, 3, 0, 1), expected=Status.BUS_TIMEOUT)
        assert len(bus.transactions) == before
        await reset(dut, start_clock=False)
        assert await exchange(pins, request(Command.HELLO, 4, 0, 0)) == [0x0401]
