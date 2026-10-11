"""Host transport checks without a simulator or a connected serial device."""

import binascii
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from rosenblatt import isa
from rosenblatt.device import Device, EmuBackend
from rosenblatt.emu import Config
from rosenblatt.uart import Command, ProtocolError, Status, UartBackend, crc16


def response(sequence, status=0, words=()):
    body = bytes([sequence, status, len(words)]) + struct.pack(f"<{len(words)}I", *words)
    return b"RB" + body + struct.pack("<H", binascii.crc_hqx(body, 0xFFFF))


class SerialPeer:
    """A framed memory peer with deliberately fragmented blocking I/O."""

    def __init__(self, fragment=3):
        self.fragment = fragment
        self.pending = bytearray()
        self.replies = bytearray()
        self.commands = []
        self.write_calls = 0
        self.transform = None
        self.caps = 0x0301  # protocol 1, at most three words per request
        self.memory = {isa.HOST_CSR_BASE + isa.CSR.HWCFG0 + i: word
                       for i, word in enumerate([0x01020408, 0x010A100A, 0x00200808])}
        self.closed = False

    def write(self, data):
        self.write_calls += 1
        n = min(len(data), self.fragment)
        self.pending.extend(data[:n])
        if len(self.pending) >= 8:
            command, sequence = self.pending[2:4]
            address = int.from_bytes(self.pending[4:7], "little")
            count = self.pending[7]
            length = 10 + (count * 4 if command == 2 else 0)
            if len(self.pending) >= length:
                packet = bytes(self.pending[:length])
                del self.pending[:length]
                assert packet[:2] == b"RB"
                assert binascii.crc_hqx(packet[2:-2], 0xFFFF) == int.from_bytes(packet[-2:], "little")
                assert count <= 3
                self.commands.append((command, sequence, address, count))
                words = []
                if command == 0:
                    words = [self.caps]
                elif command == 1:
                    words = [self.memory.get(address + i, 0) for i in range(count)]
                else:
                    for i, word in enumerate(struct.unpack(f"<{count}I", packet[8:-2])):
                        self.memory[address + i] = word
                reply = response(sequence, words=words)
                self.replies.extend(self.transform(reply) if self.transform else reply)
        return n

    def read(self, n):
        n = min(n, self.fragment, len(self.replies))
        data = bytes(self.replies[:n])
        del self.replies[:n]
        return data

    def close(self):
        self.closed = True


def test_crc_known_check_value():
    assert crc16(b"123456789") == 0x29B1


@pytest.mark.parametrize("fragment", [1, 3, 4096])
def test_discovery_and_unaligned_burst_roundtrip(fragment):
    peer = SerialPeer(fragment)
    backend = UartBackend(stream=peer)
    assert backend.cfg == Config()
    assert backend.max_words == 3
    dev = Device(backend)
    original = bytes([0xA5]) * 128
    dev.write(0, original)
    data = bytes(range(97))
    dev.write(3, data)
    expected = original[:3] + data + original[100:]
    assert dev.read(0, 128) == expected
    assert dev.read(7, 61) == expected[7:68]
    backend.bus_write(50, [-1, 1 << 32])
    assert backend.bus_read(50, 2) == [0xFFFFFFFF, 0]
    n = len(peer.commands)
    assert backend.bus_read(0, 0) == []
    backend.bus_write(0, [])
    assert len(peer.commands) == n
    backend.close()
    assert not peer.closed  # injected streams belong to their caller


def test_sequence_wraps_and_stays_in_order():
    peer = SerialPeer(4096)
    backend = UartBackend(stream=peer)
    for _ in range(260):
        backend.bus_read(0, 1)
    assert [cmd[1] for cmd in peer.commands] == [i % 256 for i in range(len(peer.commands))]


@pytest.mark.parametrize("transform,exception,match", [
    (lambda r: r[:-1] + bytes([r[-1] ^ 1]), ProtocolError, "CRC"),
    (lambda r: b"XX" + r[2:], ProtocolError, "framing"),
    (lambda r: response((r[2] + 1) % 256), ProtocolError, "sequence"),
    (lambda r: response(r[2], Status.BUS_TIMEOUT), ProtocolError, "BUS_TIMEOUT"),
    (lambda r: response(r[2], 255), ProtocolError, "255"),
    (lambda r: response(r[2], words=[123]), ProtocolError, "expected 0"),
    (lambda r: r[:-1], TimeoutError, "timed out"),
    (lambda r: b"", TimeoutError, "timed out"),
])
def test_failed_write_is_never_retried(transform, exception, match):
    peer = SerialPeer()
    backend = UartBackend(stream=peer)
    peer.transform = transform
    before = len(peer.commands)
    with pytest.raises(exception, match=match):
        backend.bus_write(12, [0x12345678])
    assert len(peer.commands) == before + 1
    # A lost/corrupt ACK does not imply that the write failed on the device.
    assert peer.memory[12] == 0x12345678
    calls = peer.write_calls
    with pytest.raises(ProtocolError, match="failed"):
        backend.bus_read(12, 1)
    assert peer.write_calls == calls


def test_zero_progress_write_fails_connection():
    peer = SerialPeer()
    backend = UartBackend(stream=peer)
    peer.fragment = 0
    with pytest.raises(TimeoutError, match="write did not complete"):
        backend.bus_read(0, 1)
    with pytest.raises(ProtocolError, match="failed"):
        backend.bus_read(0, 1)


@pytest.mark.parametrize("addr,count", [(-1, 1), (0, -1), (1 << 24, 1), ((1 << 24) - 1, 2)])
def test_invalid_word_range_sends_nothing(addr, count):
    peer = SerialPeer()
    backend = UartBackend(stream=peer)
    calls = peer.write_calls
    with pytest.raises(ValueError, match="24-bit"):
        backend.bus_read(addr, count)
    assert peer.write_calls == calls


def test_device_memory_bounds_and_closed_connection():
    peer = SerialPeer()
    backend = UartBackend(stream=peer)
    calls = peer.write_calls
    with pytest.raises(ValueError, match="global memory"):
        backend.write_bytes(backend.cfg.gmem_bytes - 1, b"ab")
    with pytest.raises(ValueError, match="global memory"):
        backend.read_bytes(0, -1)
    assert peer.write_calls == calls
    backend.close()
    with pytest.raises(ProtocolError, match="closed"):
        backend.bus_read(0, 1)


def test_kernel_timeout_fails_connection(monkeypatch):
    peer = SerialPeer()
    backend = UartBackend(stream=peer)
    peer.memory[isa.HOST_CSR_BASE + isa.CSR.CTRL] = 1
    # Advance a synthetic clock on each read so this does not sleep or race.
    now = [0.0]
    original_read = backend.bus_read

    def read(*args):
        result = original_read(*args)
        now[0] += 0.1
        return result

    monkeypatch.setattr(backend, "bus_read", read)
    monkeypatch.setattr("rosenblatt.device.time.monotonic", lambda: now[0])
    with pytest.raises(TimeoutError, match="kernel"):
        backend.wait_idle(0.25)
    with pytest.raises(ProtocolError, match="failed"):
        backend.bus_read(0, 1)


@pytest.mark.parametrize("caps", [0x0301, 0x0302, 1])
def test_owned_serial_port_lifecycle(monkeypatch, caps):
    peer = SerialPeer()
    peer.caps = caps
    options = {}

    def open_serial(**kwargs):
        options.update(kwargs)
        return peer

    monkeypatch.setitem(sys.modules, "serial", SimpleNamespace(Serial=open_serial))
    if caps == 0x0301:
        with UartBackend("TEST", baudrate=230400):
            assert not peer.closed
    else:
        with pytest.raises(ProtocolError, match="capabilities"):
            UartBackend("TEST", baudrate=230400)
    assert peer.closed
    assert options == dict(port="TEST", baudrate=230400, timeout=2.0, write_timeout=2.0,
                           bytesize=8, parity="N", stopbits=1)


def test_demo_selects_discovered_scalar_core_and_closes_port(monkeypatch, capsys):
    from ml import demo

    backend = EmuBackend(Config(dp4a=False))
    closed = []

    class Connection:
        def __enter__(self):
            return backend

        def __exit__(self, *_):
            closed.append(True)

    monkeypatch.setattr(demo, "UartBackend", lambda port, **kwargs: Connection())
    monkeypatch.setattr(demo.mnist, "load", lambda split: {
        "test_images": np.zeros((1, 28, 28), dtype=np.uint8), "test_labels": np.array([0]),
    })
    assert demo.main(["--port", "TEST", "--count", "1"]) == 0
    assert closed == [True]
    assert "serial GPU (TEST): scalar int8" in capsys.readouterr().out
