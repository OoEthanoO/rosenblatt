"""Framed 8N1 serial transport for rb_serial_gpu (see docs/serial.md).

Only one request may be in flight. Writes are acknowledged after their bus
handshakes; failed transactions are never retried automatically. A failed
connection must be closed and the board reset before reconnecting.
"""

from __future__ import annotations

import binascii
import operator
import struct
import time
from enum import IntEnum

from . import isa
from .device import BusBackend

MAGIC = b"RB"
PROTOCOL_VERSION = 1


class Command(IntEnum):
    HELLO = 0
    READ = 1
    WRITE = 2


class Status(IntEnum):
    OK = 0
    BAD_COMMAND = 1
    BAD_LENGTH = 2
    BAD_CRC = 3
    BUS_TIMEOUT = 4
    BAD_ADDRESS = 5


class ProtocolError(RuntimeError):
    pass


def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE: poly 0x1021, init 0xffff, no reflection/xorout."""
    return binascii.crc_hqx(data, 0xFFFF)


def request(command: int, sequence: int, address: int, count: int, payload: bytes = b"") -> bytes:
    body = bytes([command, sequence]) + address.to_bytes(3, "little") + bytes([count]) + payload
    return MAGIC + body + struct.pack("<H", crc16(body))


class UartBackend(BusBackend):
    """Connect a Device to a serial GPU, discovering burst limits and hardware.

    `stream` is an optional blocking serial-like object for testing; its read,
    write, timeout and write_timeout follow pyserial. Injected streams remain
    owned by the caller. This backend is intended for one host thread.
    """

    def __init__(self, port: str | None = None, *, baudrate: int = 115200,
                 timeout: float = 2.0, kernel_timeout: float = 60.0, stream=None):
        if timeout <= 0 or kernel_timeout <= 0 or baudrate <= 0:
            raise ValueError("baudrate and timeouts must be positive")
        if stream is not None and port is not None:
            raise ValueError("supply a port or a stream, not both")
        self.timeout = timeout
        self.kernel_timeout = kernel_timeout
        self._sequence = 0
        self._failed = False
        self._closed = False
        self._owns_stream = stream is None
        if stream is None:
            if not port:
                raise ValueError("a serial port is required")
            try:
                import serial
            except ImportError as exc:
                raise RuntimeError('serial support requires: python -m pip install -e ".[fpga]"') from exc
            stream = serial.Serial(port=port, baudrate=baudrate, timeout=timeout,
                                   write_timeout=timeout, bytesize=8, parity="N", stopbits=1)
        self.stream = stream
        try:
            caps, = self._exchange(Command.HELLO, 0, 0)
            self.max_words = (caps >> 8) & 0xFF
            if caps & 0xFF != PROTOCOL_VERSION or caps >> 16 or self.max_words == 0:
                raise ProtocolError(f"unsupported serial capabilities: {caps:#x}")
            version = self.bus_read(isa.HOST_CSR_BASE + isa.CSR.HWCFG0, 1)[0] >> 24
            if version != isa.HW_VERSION:
                raise ProtocolError(f"unsupported GPU hardware version: {version}")
            self.cfg = self.read_config()
            if min(self.cfg.num_cores, self.cfg.num_lanes, self.cfg.num_warps) <= 0:
                raise ProtocolError("GPU reported an empty hardware configuration")
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if not self._closed and self._owns_stream:
            self.stream.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("serial transaction timed out; reset the board before reconnecting")
        return remaining

    def _read_exact(self, count: int, deadline: float) -> bytes:
        data = bytearray()
        while len(data) < count:
            self.stream.timeout = self._remaining(deadline)
            chunk = self.stream.read(count - len(data))
            if not chunk:
                raise TimeoutError("serial response timed out; reset the board before reconnecting")
            data.extend(chunk)
        return bytes(data)

    def _exchange(self, command: Command, address: int, count: int, payload: bytes = b"") -> list[int]:
        if self._closed or self._failed:
            raise ProtocolError("serial connection is closed or failed; reset the board and reconnect")
        sequence = self._sequence
        self._sequence = (sequence + 1) & 0xFF
        packet = request(command, sequence, address, count, payload)
        deadline = time.monotonic() + self.timeout
        try:
            sent = 0
            while sent < len(packet):
                self.stream.write_timeout = self._remaining(deadline)
                n = self.stream.write(packet[sent:])
                if not n or not 0 < n <= len(packet) - sent:
                    raise TimeoutError("serial write did not complete; reset the board before reconnecting")
                sent += n
            header = self._read_exact(5, deadline)
            if header[:2] != MAGIC or header[2] != sequence:
                raise ProtocolError("serial response has incorrect framing or sequence")
            status, size = header[3:5]
            tail = self._read_exact(size * 4 + 2, deadline)
            body = header[2:] + tail[:-2]
            if crc16(body) != int.from_bytes(tail[-2:], "little"):
                raise ProtocolError("serial response CRC mismatch")
            if status != Status.OK:
                name = Status(status).name if status in Status._value2member_map_ else str(status)
                raise ProtocolError(f"serial bridge rejected request: {name}")
            expected = 1 if command == Command.HELLO else count if command == Command.READ else 0
            if size != expected:
                raise ProtocolError(f"serial response has {size} words, expected {expected}")
            return list(struct.unpack(f"<{size}I", tail[:-2]))
        except Exception:
            # The board may have performed a write even when its reply was lost.
            self._failed = True
            raise

    @staticmethod
    def _word_range(address: int, count: int) -> tuple[int, int]:
        address, count = operator.index(address), operator.index(count)
        if address < 0 or count < 0 or address >= 1 << 24 or address + count > 1 << 24:
            raise ValueError("serial bus range is outside the 24-bit word address space")
        return address, count

    def bus_write(self, word_addr: int, words: list[int]) -> None:
        word_addr, count = self._word_range(word_addr, len(words))
        for start in range(0, count, self.max_words):
            chunk = [operator.index(w) & isa.MASK32 for w in words[start:start + self.max_words]]
            payload = struct.pack(f"<{len(chunk)}I", *chunk)
            self._exchange(Command.WRITE, word_addr + start, len(chunk), payload)

    def bus_read(self, word_addr: int, n: int) -> list[int]:
        word_addr, n = self._word_range(word_addr, n)
        result = []
        for start in range(0, n, self.max_words):
            result.extend(self._exchange(Command.READ, word_addr + start, min(n - start, self.max_words)))
        return result

    def _byte_range(self, address: int, count: int) -> None:
        address, count = operator.index(address), operator.index(count)
        if address < 0 or count < 0 or address + count > self.cfg.gmem_bytes:
            raise ValueError("transfer is outside device global memory")

    def write_bytes(self, addr: int, data: bytes) -> None:
        data = bytes(data)
        self._byte_range(addr, len(data))
        super().write_bytes(addr, data)

    def read_bytes(self, addr: int, n: int) -> bytes:
        self._byte_range(addr, n)
        return super().read_bytes(addr, n)

    def wait_idle(self, timeout_s: float | None = None) -> None:
        try:
            super().wait_idle(self.kernel_timeout if timeout_s is None else timeout_s)
        except TimeoutError:
            self._failed = True
            raise
