"""Functional emulator: the golden model the RTL is checked against.

It executes kernels instruction by instruction with exactly the architectural
semantics of the hardware: lockstep warps, per-thread active masks, the SSY/SYNC
divergence stack, barriers, wrap-around memory addressing, and "highest lane
wins" when several lanes of one store hit the same byte.

It is not cycle accurate (blocks run one after another, warps take turns one
instruction at a time) but it does count the events that decide performance on
the real design: instructions, active lanes, coalesced global-memory requests,
shared-memory bank wavefronts and matrix-unit steps.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from . import isa
from .isa import Fn, Op, SR, MASK32

READY, AT_BARRIER, DONE = 0, 1, 2


class EmuError(RuntimeError):
    pass


@dataclass
class Config:
    """Device configuration. The same object parameterizes the RTL build."""

    num_cores: int = 2
    num_lanes: int = 8          # warp width = SIMD lanes per core
    num_warps: int = 4          # warp slots per core; max threads per block = lanes * warps
    imem_words: int = 1024
    gmem_bytes: int = 256 * 1024
    smem_bytes: int = 4 * 1024  # per core
    stack_depth: int = 8
    mul_width: int = 32         # 16 = only the low 16 bits of each operand are multiplied
    dp4a: bool = True
    gmem_latency: int = 8       # cycles; timing only, ignored by the emulator
    gmem_interval: int = 1      # cycles between accepted requests; timing only
    lsu_depth: int = 4          # global memory instructions in flight per core; timing only
    park: bool = False          # warps sleep (instead of retrying) while the LSU is busy; timing only
    gmem_line_bytes: int = 4    # bytes per global-memory load (4, 8, 16 or 32): lanes in one line coalesce; stores coalesce per word
    scoreboard: bool = False    # a global load stalls only instructions that use its register (else the warp); timing only
    mma: bool = False           # matrix unit: mma, mmaz, mmar (needs dp4a)
    mma_slots: int = 1          # mma/mmaz per warp in the matrix unit at once (a power of two); timing only

    @property
    def max_threads(self) -> int:
        return self.num_lanes * self.num_warps

    def verilog_params(self) -> dict[str, int]:
        lg = lambda n: (n - 1).bit_length()  # noqa: E731
        return {
            "NUM_CORES": self.num_cores,
            "NUM_LANES": self.num_lanes,
            "NUM_WARPS": self.num_warps,
            "IMEM_AW": lg(self.imem_words),
            "GMEM_AW": lg(self.gmem_bytes // 4),
            "SMEM_AW": lg(self.smem_bytes // 4),
            "STACK_DEPTH": self.stack_depth,
            "MUL_WIDTH": self.mul_width,
            "ENABLE_DP4A": int(self.dp4a),
            "GMEM_LATENCY": self.gmem_latency,
            "GMEM_INTERVAL": self.gmem_interval,
            "LSU_DEPTH": self.lsu_depth,
            "PARK": int(self.park),
            "GMEM_LINE": self.gmem_line_bytes,
            "SCOREBOARD": int(self.scoreboard),
            "ENABLE_MMA": int(self.mma),
            "MMA_SLOTS": self.mma_slots,
        }


def _s32(x: int) -> int:
    return x - (1 << 32) if x & 0x8000_0000 else x


def _s8(x: int) -> int:
    x &= 0xFF
    return x - 256 if x & 0x80 else x


def _s16(x: int) -> int:
    x &= 0xFFFF
    return x - 65536 if x & 0x8000 else x


@dataclass
class Warp:
    wid: int
    pc: int
    active: int
    regs: list[list[int]]
    exited: int = 0
    stack: list[tuple[int, int]] = field(default_factory=list)
    state: int = READY
    unwind: bool = False
    acc: list[list[int]] = field(default_factory=list)   # mma tile: acc[lane][row], undefined until mmaz


class Emulator:
    def __init__(self, cfg: Config | None = None, strict: bool = True):
        self.cfg = cfg or Config()
        self.strict = strict
        self.gmem = bytearray(self.cfg.gmem_bytes)
        self.imem = [0] * self.cfg.imem_words
        self.stats: Counter = Counter()
        self.errors: list[str] = []
        self.max_steps = 50_000_000

    # ---------------------------------------------------------- host side
    def load_program(self, words: list[int]) -> None:
        if len(words) > len(self.imem):
            raise EmuError(f"program has {len(words)} words, imem holds {len(self.imem)}")
        self.imem[: len(words)] = [w & MASK32 for w in words]
        self.imem[len(words):] = [0] * (len(self.imem) - len(words))

    def write_bytes(self, addr: int, data: bytes) -> None:
        if addr < 0 or addr + len(data) > len(self.gmem):
            raise EmuError("write outside global memory")
        self.gmem[addr: addr + len(data)] = data

    def read_bytes(self, addr: int, n: int) -> bytes:
        if addr < 0 or addr + n > len(self.gmem):
            raise EmuError("read outside global memory")
        return bytes(self.gmem[addr: addr + n])

    # ------------------------------------------------------------ launch
    def launch(self, grid: int, block: int, args: list[int] | tuple[int, ...] = (), entry: int = 0) -> Counter:
        cfg = self.cfg
        if not 0 < block <= cfg.max_threads:
            raise EmuError(f"block size {block} not in 1..{cfg.max_threads}")
        if len(args) > isa.NUM_ARGS:
            raise EmuError("too many kernel arguments")
        self.args = [a & MASK32 for a in args] + [0] * (isa.NUM_ARGS - len(args))
        self.stats = Counter()
        self.errors = []
        self._code = [isa.decode(w) for w in self.imem]
        self._entry = entry % len(self.imem)
        for ctaid in range(grid):
            self._run_block(ctaid, block, grid, core_id=ctaid % cfg.num_cores)
        self.stats["blocks"] = grid
        return self.stats

    def _error(self, msg: str) -> None:
        self.errors.append(msg)
        if self.strict:
            raise EmuError(msg)

    def _run_block(self, ctaid: int, ntid: int, nctaid: int, core_id: int) -> None:
        cfg = self.cfg
        W = cfg.num_lanes
        nwarps = (ntid + W - 1) // W
        warps = []
        for w in range(nwarps):
            mask = 0
            for lane in range(W):
                if w * W + lane < ntid:
                    mask |= 1 << lane
            warps.append(Warp(w, self._entry, mask, [[0] * isa.NUM_REGS for _ in range(W)],
                              acc=[[0] * W for _ in range(W)]))
        self._smem = bytearray(cfg.smem_bytes)
        self._block = (ctaid, ntid, nctaid, core_id)
        steps = 0
        while True:
            progressed = False
            for warp in warps:
                if warp.state != READY:
                    continue
                self._step(warp)
                progressed = True
                steps += 1
            live = [w for w in warps if w.state != DONE]
            if not live:
                return
            if all(w.state == AT_BARRIER for w in live):
                for w in live:
                    w.state = READY
                self.stats["barriers"] += 1
            elif not progressed:
                raise EmuError("deadlock: no warp can make progress")
            if steps > self.max_steps:
                raise EmuError(f"block {ctaid} exceeded {self.max_steps} instructions (infinite loop?)")

    # ------------------------------------------------------- divergence
    def _push(self, warp: Warp, pc: int, mask: int) -> None:
        if len(warp.stack) >= self.cfg.stack_depth:
            self._error(f"warp {warp.wid}: divergence stack overflow at pc {warp.pc}")
            return
        warp.stack.append((pc, mask))

    def _pop(self, warp: Warp) -> None:
        if not warp.stack:
            self._error(f"warp {warp.wid}: sync with an empty stack at pc {warp.pc}")
            warp.pc += 1
            return
        pc, mask = warp.stack.pop()
        warp.pc = pc
        warp.active = mask & ~warp.exited
        self._settle(warp)

    def _settle(self, warp: Warp) -> None:
        """If no thread is active any more, finish the warp or unwind the stack."""
        if warp.active == 0:
            if warp.stack:
                warp.unwind = True
            else:
                warp.state = DONE

    # ------------------------------------------------------ one instruction
    def _step(self, warp: Warp) -> None:
        cfg, st = self.cfg, self.stats
        if warp.unwind:
            warp.unwind = False
            st["unwinds"] += 1
            self._pop(warp)
            return
        imem_n = len(self._code)
        d = self._code[warp.pc % imem_n]
        op = d.opcode
        M = warp.active
        lanes = [lane for lane in range(cfg.num_lanes) if M >> lane & 1]
        R = warp.regs
        st["instructions"] += 1
        st["lane_ops"] += len(lanes)
        group = op >> 4

        if group <= 1:  # ------------------------------------------- ALU
            fn = op & 0xF
            if fn == Fn.DP4A and not cfg.dp4a:
                self._error(f"dp4a is not enabled in this configuration (pc {warp.pc})")
                warp.pc += 1
                return
            rd, rs1 = d.rd, d.rs1
            if group == 0:
                rs2 = d.rs2
                bs = None
            else:
                bs = d.imm18 & MASK32
            for lane in lanes:
                r = R[lane]
                a = r[rs1]
                b = r[rs2] if bs is None else bs
                res = self._alu(fn, a, b, r[rd])
                if rd:
                    r[rd] = res
            warp.pc += 1
            return

        if group == 2:  # ---------------------------------------- memory
            self._memory(warp, d, lanes)
            warp.pc += 1
            return

        if op == Op.DP4AB:  # ------------------- dp4a, rs1 from one lane
            if not cfg.dp4a:
                self._error(f"dp4ab needs dp4a, which is not enabled (pc {warp.pc})")
            else:
                a = R[d.lane % cfg.num_lanes][d.rs1]   # read before any lane writes
                for lane in lanes:
                    r = R[lane]
                    res = self._alu(Fn.DP4A, a, r[d.rs2], r[d.rd])
                    if d.rd:
                        r[d.rd] = res
            warp.pc += 1
            return

        if op in (Op.MMA, Op.MMAZ, Op.MMAR):  # ------------------ matrix unit
            self._matrix(warp, d, lanes)
            warp.pc += 1
            return

        # -------------------------------------------------------- control
        if op == Op.LI or op == Op.LUI:
            v = (d.imm22 if op == Op.LI else d.imm22 << 10) & MASK32
            if d.rd:
                for lane in lanes:
                    R[lane][d.rd] = v
            warp.pc += 1
        elif op == Op.S2R:
            if d.rd:
                for lane in lanes:
                    R[lane][d.rd] = self._special(warp, lane, d.imm18 & 0x1F)
            warp.pc += 1
        elif op == Op.BZ or op == Op.BNZ:
            cond = 0
            for lane in lanes:
                if R[lane][d.rs1] != 0:
                    cond |= 1 << lane
            taken = M & (cond if op == Op.BNZ else ~cond)
            not_taken = M & ~taken
            target = (d.imm18 & 0x3FFFF) % imem_n
            if taken == 0:
                warp.pc += 1
            elif not_taken == 0:
                warp.pc = target
            else:  # divergence: run the fall-through path now, defer the taken one
                st["divergent_branches"] += 1
                self._push(warp, target, taken)
                warp.active = not_taken
                warp.pc += 1
        elif op == Op.SSY:
            self._push(warp, (d.imm18 & 0x3FFFF) % imem_n, M)
            warp.pc += 1
        elif op == Op.SYNC:
            self._pop(warp)
        elif op == Op.BAR:
            warp.state = AT_BARRIER
            warp.pc += 1
        elif op == Op.RET:
            warp.exited |= M
            warp.active = 0
            self._settle(warp)
        else:
            self._error(f"illegal opcode 0x{op:02x} at pc {warp.pc}")
            warp.pc += 1

    def _matrix(self, warp: Warp, d: isa.Decoded, lanes: list[int]) -> None:
        """The warp's accumulator tile C (W x W int32; lane i holds column i).

        mma:  C[j][i] += dot4(lane j's rs1, lane i's rs2), for every row j and
              every active lane i: an (W x 4) x (4 x W) int8 matrix product
        mmaz: C[j][i] = 0 for the active lanes i
        mmar: rd = C[row][i], row taken modulo W
        """
        cfg, W = self.cfg, self.cfg.num_lanes
        if not (cfg.mma and cfg.dp4a):
            self._error(f"{isa.BY_OPCODE[d.opcode].mnemonic} needs the matrix unit (mma and dp4a), "
                        f"which is not enabled (pc {warp.pc})")
            return
        R = warp.regs
        if d.opcode == Op.MMAR:
            if d.rd:
                for lane in lanes:
                    R[lane][d.rd] = warp.acc[lane][d.lane % W]
            return
        self.stats["mma_steps"] += W
        if d.opcode == Op.MMAZ:
            for lane in lanes:
                warp.acc[lane] = [0] * W
            return
        a = [R[j][d.rs1] for j in range(W)]   # row j of A: lane j's rs1, whether or not it is active
        for lane in lanes:
            b, col = R[lane][d.rs2], warp.acc[lane]
            for j in range(W):
                col[j] = self._alu(Fn.DP4A, a[j], b, col[j])

    def _alu(self, fn: int, a: int, b: int, c: int) -> int:
        if fn == Fn.ADD:
            r = a + b
        elif fn == Fn.SUB:
            r = a - b
        elif fn == Fn.MUL:
            r = self._mul(a, b)
        elif fn == Fn.MAD:
            r = self._mul(a, b) + c
        elif fn == Fn.AND:
            r = a & b
        elif fn == Fn.OR:
            r = a | b
        elif fn == Fn.XOR:
            r = a ^ b
        elif fn == Fn.SHL:
            r = a << (b & 31)
        elif fn == Fn.SHR:
            r = a >> (b & 31)
        elif fn == Fn.SRA:
            r = _s32(a) >> (b & 31)
        elif fn == Fn.SLT:
            r = int(_s32(a) < _s32(b))
        elif fn == Fn.SLTU:
            r = int(a < b)
        elif fn == Fn.SEQ:
            r = int(a == b)
        elif fn == Fn.MIN:
            r = min(_s32(a), _s32(b))
        elif fn == Fn.MAX:
            r = max(_s32(a), _s32(b))
        else:  # DP4A
            r = c + sum(_s8(a >> s) * _s8(b >> s) for s in (0, 8, 16, 24))
        return r & MASK32

    def _mul(self, a: int, b: int) -> int:
        if self.cfg.mul_width == 16:
            return _s16(a) * _s16(b)
        return a * b

    def _special(self, warp: Warp, lane: int, idx: int) -> int:
        ctaid, ntid, nctaid, core_id = self._block
        W = self.cfg.num_lanes
        tid = warp.wid * W + lane
        if idx == SR.TID:
            return tid
        if idx == SR.NTID:
            return ntid
        if idx == SR.CTAID:
            return ctaid
        if idx == SR.NCTAID:
            return nctaid
        if idx == SR.LANEID:
            return lane
        if idx == SR.WARPID:
            return warp.wid
        if idx == SR.COREID:
            return core_id
        if idx == SR.CLOCK:
            return self.stats["instructions"] & MASK32
        if SR.ARG0 <= idx <= SR.ARG7:
            return self.args[idx - SR.ARG0]
        if idx == SR.GTID:
            return (ctaid * ntid + tid) & MASK32
        return 0

    def _memory(self, warp: Warp, d: isa.Decoded, lanes: list[int]) -> None:
        cfg, st = self.cfg, self.stats
        op = d.opcode
        shared = bool(op & isa.MEM_SHARED_BIT)
        store = bool(op & isa.MEM_STORE_BIT)
        size = op & 3
        if size == 3 or (store and size == isa.MEM_SIZE_UBYTE):
            self._error(f"illegal memory opcode 0x{op:02x} at pc {warp.pc}")
            return
        mem = self._smem if shared else self.gmem
        nwords = len(mem) // 4
        R = warp.regs
        addrs = []
        for lane in lanes:
            a = (R[lane][d.rs1] + d.imm18) & MASK32
            word = (a >> 2) % nwords
            addrs.append((lane, word, a & 3))

        # performance accounting, mirroring the hardware's coalescer / bank logic
        if shared:
            # Each bank serves one row per wavefront: loads of the same word share
            # it (broadcast); stores are written one lane per bank per wavefront.
            st["smem_instr"] += 1
            banks: dict[int, list[int]] = {}
            for _, word, _ in addrs:
                banks.setdefault(word % cfg.num_lanes, []).append(word // cfg.num_lanes)
            per_bank = (len(rows) if store else len(set(rows)) for rows in banks.values())
            st["smem_wavefronts"] += max(per_bank, default=0)
        else:
            st["gmem_instr"] += 1
            st["gmem_lanes"] += len(addrs)
            # loads: one request per distinct line; stores coalesce only within a word
            words_per_request = 1 if store else cfg.gmem_line_bytes // 4
            st["gmem_reqs"] += len({word // words_per_request for _, word, _ in addrs})
            st["gmem_loads" if not store else "gmem_stores"] += 1

        if store:
            for lane, word, off in addrs:  # ascending lane order: highest lane wins
                v = R[lane][d.rd]
                base = word * 4
                if size == isa.MEM_SIZE_WORD:
                    mem[base: base + 4] = v.to_bytes(4, "little")
                else:
                    mem[base + off] = v & 0xFF
            return
        if d.rd == 0:
            return
        for lane, word, off in addrs:
            base = word * 4
            if size == isa.MEM_SIZE_WORD:
                v = int.from_bytes(mem[base: base + 4], "little")
            elif size == isa.MEM_SIZE_BYTE:
                v = _s8(mem[base + off]) & MASK32
            else:
                v = mem[base + off]
            R[lane][d.rd] = v
