"""Host-side runtime: the "CUDA driver" of the Rosenblatt GPU.

    dev = Device(EmuBackend())          # or a cocotb RTL backend, or UartBackend(port)
    x   = dev.upload(np.arange(64, dtype=np.int32))
    y   = dev.alloc(64 * 4)
    dev.load(assemble(src))
    run = dev.launch(grid=2, block=32, args=[x, y, 64])
    print(dev.download(y, 64, np.int32), run.cycles)

The same calls work on every backend, so a kernel is developed on the
emulator, verified against the RTL in simulation and then run on the FPGA
without changing a line.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import isa
from .asm import Program
from .emu import Config, Emulator
from .isa import CSR, Perf


@dataclass
class LaunchResult:
    grid: int
    block: int
    cycles: int | None = None              # None on the emulator (it is not cycle accurate)
    gmem_reqs: int | None = None           # global-memory requests (device-wide)
    errors: int = 0
    per_core: list[dict[str, int]] = field(default_factory=list)  # perf counters
    stats: dict[str, int] = field(default_factory=dict)          # summed/derived counters

    def total(self, name: str) -> int:
        return sum(c[name] for c in self.per_core)


class Backend:
    cfg: Config

    def write_bytes(self, addr: int, data: bytes) -> None: raise NotImplementedError
    def read_bytes(self, addr: int, n: int) -> bytes: raise NotImplementedError
    def load_program(self, words: list[int]) -> None: raise NotImplementedError
    def launch(self, grid: int, block: int, args: list[int], entry: int = 0) -> LaunchResult: raise NotImplementedError


class EmuBackend(Backend):
    def __init__(self, cfg: Config | None = None, strict: bool = True):
        self.emu = Emulator(cfg, strict=strict)
        self.cfg = self.emu.cfg

    def write_bytes(self, addr, data):
        self.emu.write_bytes(addr, data)

    def read_bytes(self, addr, n):
        return self.emu.read_bytes(addr, n)

    def load_program(self, words):
        self.emu.load_program(words)

    def launch(self, grid, block, args, entry=0):
        st = self.emu.launch(grid, block, args, entry)
        res = LaunchResult(grid, block, errors=len(self.emu.errors))
        res.stats = dict(st)
        return res


class BusBackend(Backend):
    """A device reached through the 32-bit host bus (RTL simulation, UART bridge)."""

    def bus_write(self, word_addr: int, words: list[int]) -> None: raise NotImplementedError
    def bus_read(self, word_addr: int, n: int) -> list[int]: raise NotImplementedError

    def wait_idle(self, timeout_s: float = 60.0) -> None:
        deadline = time.monotonic() + timeout_s
        while self.bus_read(isa.HOST_CSR_BASE + CSR.CTRL, 1)[0] & 1:
            if time.monotonic() > deadline:
                raise TimeoutError("kernel did not finish")

    # ---- configuration discovery
    def read_config(self) -> Config:
        c0, c1, c2, c3 = self.bus_read(isa.HOST_CSR_BASE + CSR.HWCFG0, 4)
        extra = {} if c3 == 0 else {"lsu_depth": c3 & 0xFF, "gmem_interval": c3 >> 8 & 0xFF,
                                    "park": bool(c3 >> 16 & 0xFF)}
        return Config(**extra,
            num_lanes=c0 & 0xFF, num_warps=c0 >> 8 & 0xFF, num_cores=c0 >> 16 & 0xFF,
            imem_words=1 << (c1 & 0xFF), gmem_bytes=4 << (c1 >> 8 & 0xFF),
            smem_bytes=4 << (c1 >> 16 & 0xFF), dp4a=bool(c1 >> 24 & isa.FEATURE_DP4A),
            gmem_latency=c2 & 0xFF, stack_depth=c2 >> 8 & 0xFF, mul_width=c2 >> 16 & 0xFF,
        )

    # ---- memory
    def write_bytes(self, addr, data):
        data = bytes(data)
        if not data:
            return
        start, end = addr & ~3, (addr + len(data) + 3) & ~3
        buf = bytearray(end - start)
        if addr != start or (addr + len(data)) != end:  # partial words: read-modify-write
            buf[0:4] = self.read_bytes(start, 4)
            buf[-4:] = self.read_bytes(end - 4, 4)
        buf[addr - start: addr - start + len(data)] = data
        words = np.frombuffer(bytes(buf), dtype="<u4").tolist()
        self.bus_write(isa.HOST_GMEM_BASE + start // 4, words)

    def read_bytes(self, addr, n):
        if n <= 0:
            return b""
        start, end = addr & ~3, (addr + n + 3) & ~3
        words = self.bus_read(isa.HOST_GMEM_BASE + start // 4, (end - start) // 4)
        raw = np.asarray(words, dtype="<u4").tobytes()
        return raw[addr - start: addr - start + n]

    def load_program(self, words):
        self.bus_write(isa.HOST_IMEM_BASE, list(words))

    def launch(self, grid, block, args, entry=0):
        csr = isa.HOST_CSR_BASE
        self.bus_write(csr + CSR.GRID, [grid, block, entry])
        argv = list(args) + [0] * (isa.NUM_ARGS - len(args))
        self.bus_write(csr + CSR.ARG0, [a & isa.MASK32 for a in argv])
        self.bus_write(csr + CSR.CTRL, [1])
        self.wait_idle()
        status, = self.bus_read(csr + CSR.CTRL, 1)
        cycles, gmem_reqs, errors = self.bus_read(csr + CSR.CYCLES, 3)
        res = LaunchResult(grid, block, cycles=cycles, gmem_reqs=gmem_reqs, errors=errors)
        for core in range(self.cfg.num_cores):
            vals = self.bus_read(csr + CSR.PERF_BASE + core * isa.PERF_STRIDE, len(Perf))
            res.per_core.append({p.name.lower(): v for p, v in zip(Perf, vals)})
        if not status & 2:
            raise RuntimeError(f"kernel did not complete (status {status:#x})")
        res.stats = _derived_stats(res)
        return res


def _derived_stats(res: LaunchResult) -> dict[str, int]:
    """Device-wide totals with the same names the emulator uses."""
    t = res.total
    return {
        "instructions": t("issued"),
        "lane_ops": t("lane_ops"),
        "gmem_instr": t("gmem_instr"),
        "gmem_reqs": t("gmem_reqs"),
        "gmem_lanes": t("gmem_lanes"),
        "smem_instr": t("smem_instr"),
        "smem_wavefronts": t("smem_cycles"),
        "divergent_branches": t("diverge"),
        "blocks": t("blocks"),
    }


class Device:
    def __init__(self, backend: Backend):
        self.backend = backend
        self.cfg = backend.cfg
        self._heap = 0
        self.last: LaunchResult | None = None
        self.program: Program | None = None

    # ---- memory management (bump allocator; free everything with reset_heap)
    def alloc(self, nbytes: int, align: int = 16) -> int:
        addr = (self._heap + align - 1) // align * align
        if addr + nbytes > self.cfg.gmem_bytes:
            raise MemoryError(f"out of device memory ({addr + nbytes} > {self.cfg.gmem_bytes})")
        self._heap = addr + nbytes
        return addr

    def reset_heap(self) -> None:
        self._heap = 0

    def write(self, addr: int, data) -> None:
        if isinstance(data, np.ndarray):
            data = np.ascontiguousarray(data).astype(data.dtype.newbyteorder("<"), copy=False).tobytes()
        self.backend.write_bytes(addr, bytes(data))

    def read(self, addr: int, nbytes: int) -> bytes:
        return self.backend.read_bytes(addr, nbytes)

    def upload(self, array: np.ndarray, align: int = 16) -> int:
        addr = self.alloc(array.nbytes, align)
        self.write(addr, array)
        return addr

    def download(self, addr: int, count: int, dtype=np.int32) -> np.ndarray:
        dt = np.dtype(dtype).newbyteorder("<")
        return np.frombuffer(self.read(addr, count * dt.itemsize), dtype=dt).astype(np.dtype(dtype))

    # ---- execution
    def load(self, program: Program | list[int]) -> None:
        words = program.words if isinstance(program, Program) else list(program)
        if len(words) > self.cfg.imem_words:
            raise ValueError(f"program ({len(words)} words) does not fit imem ({self.cfg.imem_words})")
        self.backend.load_program(words)
        self.program = program if isinstance(program, Program) else None

    def launch(self, grid: int, block: int, args=(), entry: int | str = 0) -> LaunchResult:
        """Run a kernel. `entry` is an instruction address or a label of the loaded program."""
        if isinstance(entry, str):
            entry = self.program.labels[entry]
        if not 0 < block <= self.cfg.max_threads:
            raise ValueError(f"block must be 1..{self.cfg.max_threads} threads, got {block}")
        if len(args) > isa.NUM_ARGS:
            raise ValueError("at most 8 kernel arguments")
        self.last = self.backend.launch(grid, block, [int(a) for a in args], entry)
        if self.last.errors:
            raise RuntimeError(f"device reported errors: {self.last.errors:#x}")
        return self.last
