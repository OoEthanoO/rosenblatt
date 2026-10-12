"""Rosenblatt ISA: the single source of truth.

The assembler, the emulator and the RTL (through the generated rtl/rb_isa.vh)
all take their encodings from this file, so they cannot drift apart.

Instruction word, 32 bits:

     31    26 25  22 21  18 17  14 13             0
    +--------+------+------+------+----------------+
    | opcode |  rd  | rs1  | rs2  |    unused      |  R-type  rd = rs1 OP rs2
    | opcode |  rd  | rs1  |     imm18 (signed)    |  I-type  rd = rs1 OP imm
    | opcode |  rd  |         imm22 (signed)       |  J-type  li / lui
    +--------+------+------+------+----------------+

The opcode's top two bits pick the group:

    00ffff  ALU, register operand    (ffff = ALU function)
    01ffff  ALU, immediate operand   (same function codes)
    10ssww  memory: s = shared?, s2 = store?, ww = 00 word / 01 byte / 10 ubyte
    11xxxx  control

Every thread has 16 32-bit registers; r0 always reads as zero, so the all-zero
word 0x00000000 (add r0, r0, r0) is a natural NOP.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

XLEN = 32
NUM_REGS = 16
MASK32 = 0xFFFF_FFFF

# ---------------------------------------------------------------- fields
OP_SHIFT, RD_SHIFT, RS1_SHIFT, RS2_SHIFT = 26, 22, 18, 14
IMM18_BITS, IMM22_BITS = 18, 22


class Fn(IntEnum):
    """ALU function codes (opcode bits [3:0] for groups 00 and 01)."""

    ADD = 0
    SUB = 1
    MUL = 2   # low 32 bits of the product
    MAD = 3   # rd = rs1 * rs2 + rd
    AND = 4
    OR = 5
    XOR = 6
    SHL = 7
    SHR = 8   # logical
    SRA = 9   # arithmetic
    SLT = 10  # signed   (result 0/1)
    SLTU = 11  # unsigned (result 0/1)
    SEQ = 12  # equal    (result 0/1)
    MIN = 13  # signed
    MAX = 14  # signed
    DP4A = 15  # rd += dot(int8x4 rs1, int8x4 rs2)   (optional in hardware)


class Op(IntEnum):
    """Non-ALU opcodes. ALU opcodes are Fn (register form) or 0x10|Fn (immediate form)."""

    # memory: 0b10 S W ss
    LDW = 0x20
    LDB = 0x21
    LDBU = 0x22
    STW = 0x24
    STB = 0x25
    LDS = 0x28
    LDSB = 0x29
    LDSBU = 0x2A
    STS = 0x2C
    STSB = 0x2D
    # control
    LI = 0x30    # rd = sext(imm22)
    LUI = 0x31   # rd = imm22 << 10
    S2R = 0x32   # rd = special register imm
    BZ = 0x34    # per-thread: branch if rs1 == 0
    BNZ = 0x35   # per-thread: branch if rs1 != 0
    SSY = 0x36   # push reconvergence point
    SYNC = 0x37  # pop: go to the next deferred path / reconverge
    BAR = 0x38   # block-wide barrier (__syncthreads)
    RET = 0x39   # thread exits


ALU_R_BASE = 0x00
ALU_I_BASE = 0x10

MEM_SHARED_BIT = 0x08
MEM_STORE_BIT = 0x04
MEM_SIZE_WORD, MEM_SIZE_BYTE, MEM_SIZE_UBYTE = 0, 1, 2


class SR(IntEnum):
    """Special registers readable with s2r (CUDA names in comments)."""

    TID = 0      # threadIdx.x
    NTID = 1     # blockDim.x
    CTAID = 2    # blockIdx.x
    NCTAID = 3   # gridDim.x
    LANEID = 4   # lane within the warp
    WARPID = 5   # warp within the block
    COREID = 6   # which core runs this block
    CLOCK = 7    # core cycle counter (low 32 bits)
    ARG0 = 8     # kernel arguments 0..7 (set by the host at launch)
    ARG1 = 9
    ARG2 = 10
    ARG3 = 11
    ARG4 = 12
    ARG5 = 13
    ARG6 = 14
    ARG7 = 15
    GTID = 16    # blockIdx.x * blockDim.x + threadIdx.x


NUM_ARGS = 8

# ------------------------------------------------------- assembly syntax
# Operand formats used by the assembler/disassembler.
#   rrr  rd, rs1, rs2        rri  rd, rs1, imm       ld  rd, [rs1 + imm]
#   st   [rs1 + imm], rd     ri   rd, imm22          sr  rd, %special
#   br   rs1, target         tgt  target             -   (no operands)


@dataclass(frozen=True)
class Spec:
    mnemonic: str
    opcode: int
    fmt: str


def _build_specs() -> list[Spec]:
    specs: list[Spec] = []
    for fn in Fn:
        specs.append(Spec(fn.name.lower(), ALU_R_BASE | fn, "rrr"))
    for fn in Fn:
        if fn == Fn.DP4A:
            continue  # no immediate form exposed
        specs.append(Spec(fn.name.lower() + "i", ALU_I_BASE | fn, "rri"))
    for op in (Op.LDW, Op.LDB, Op.LDBU, Op.LDS, Op.LDSB, Op.LDSBU):
        specs.append(Spec(op.name.lower(), op, "ld"))
    for op in (Op.STW, Op.STB, Op.STS, Op.STSB):
        specs.append(Spec(op.name.lower(), op, "st"))
    specs += [
        Spec("li", Op.LI, "ri"),
        Spec("lui", Op.LUI, "ri"),
        Spec("s2r", Op.S2R, "sr"),
        Spec("bz", Op.BZ, "br"),
        Spec("bnz", Op.BNZ, "br"),
        Spec("ssy", Op.SSY, "tgt"),
        Spec("sync", Op.SYNC, "-"),
        Spec("bar", Op.BAR, "-"),
        Spec("ret", Op.RET, "-"),
    ]
    return specs


SPECS = _build_specs()
BY_MNEMONIC = {s.mnemonic: s for s in SPECS}
BY_OPCODE = {s.opcode: s for s in SPECS}

# ---------------------------------------------------- host address map
# The host (testbench, UART bridge) talks to the device through a 32-bit word bus.
# Addresses below are *word* addresses on that bus.
HOST_GMEM_BASE = 0x000000   # global memory, word i = bytes 4i..4i+3
HOST_IMEM_BASE = 0x100000   # instruction memory (write-only, broadcast to all cores)
HOST_CSR_BASE = 0x200000    # control / status registers
HOST_BOARD_BASE = 0x300000  # board extras outside the GPU: word 0 = LEDs (FPGA builds only)


class CSR(IntEnum):
    """Control/status registers, offsets from HOST_CSR_BASE."""

    CTRL = 0x00       # write 1: launch kernel.  read: bit0 busy, bit1 done, bit2 error
    GRID = 0x01       # number of blocks (gridDim.x)
    BLOCK = 0x02      # threads per block (blockDim.x)
    ENTRY = 0x03      # instruction address where the kernel starts
    ARG0 = 0x08       # 0x08..0x0F kernel arguments
    CYCLES = 0x10     # cycles taken by the last kernel
    GMEM_REQS = 0x11  # global-memory requests served during the last kernel (cores only)
    ERRORS = 0x12     # per-core sticky error bits
    HWCFG0 = 0x18     # {lanes[7:0], warps[15:8], cores[23:16], version[31:24]}
    HWCFG1 = 0x19     # {imem_aw[7:0], gmem_aw[15:8], smem_aw[23:16], features[31:24]}
    HWCFG2 = 0x1A     # {gmem_latency[7:0], stack_depth[15:8], mul_width[23:16]}
    HWCFG3 = 0x1B     # {lsu_depth[7:0], gmem_interval[15:8], park[23:16], line bytes[31:24]}  (0: not reported)
    PERF_BASE = 0x40  # per-core counters: PERF_BASE + core * PERF_STRIDE + counter


PERF_STRIDE = 0x10


class Perf(IntEnum):
    """Per-core performance counters (reset at every launch)."""

    ACTIVE = 0         # cycles the core had a block resident
    ISSUED = 1         # instructions committed (excluding replays)
    LANE_OPS = 2       # sum over committed instructions of active lanes
    IDLE = 3           # cycles in which no instruction issued
    IDLE_MEM = 4       # ...of which at least one warp was waiting on memory
    REPLAYS = 5        # re-issues: LSU busy, or a shared-memory bank conflict
    GMEM_INSTR = 6     # global load/store instructions executed
    GMEM_REQS = 7      # 32-bit requests sent to global memory (after coalescing)
    GMEM_LANES = 8     # lane accesses inside those instructions (before coalescing)
    SMEM_INSTR = 9     # shared-memory instructions executed
    SMEM_CYCLES = 10   # shared-memory wavefronts (> SMEM_INSTR means bank conflicts)
    LSU_BUSY = 11      # cycles the global load/store unit was occupied
    DIVERGE = 12       # branches that split a warp
    BLOCKS = 13        # blocks executed on this core


FEATURE_DP4A = 0x01
HW_VERSION = 1

# --------------------------------------------------------- encode/decode


def sext(value: int, bits: int) -> int:
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >> (bits - 1) else value


def fits_signed(value: int, bits: int) -> bool:
    return -(1 << (bits - 1)) <= value < (1 << (bits - 1))


def encode(opcode: int, rd: int = 0, rs1: int = 0, rs2: int = 0, imm: int | None = None) -> int:
    """Pack one instruction. imm goes into imm18, or imm22 for li/lui."""
    for name, r in (("rd", rd), ("rs1", rs1), ("rs2", rs2)):
        if not 0 <= r < NUM_REGS:
            raise ValueError(f"{name}={r} is not a register")
    word = (opcode & 0x3F) << OP_SHIFT | rd << RD_SHIFT
    if opcode in (Op.LI, Op.LUI):
        if imm is None or not fits_signed(imm, IMM22_BITS):
            raise ValueError(f"imm22 out of range: {imm}")
        return word | (imm & ((1 << IMM22_BITS) - 1))
    word |= rs1 << RS1_SHIFT
    if imm is not None:
        if not fits_signed(imm, IMM18_BITS):
            raise ValueError(f"imm18 out of range: {imm}")
        return word | (imm & ((1 << IMM18_BITS) - 1))
    return word | rs2 << RS2_SHIFT


@dataclass(frozen=True)
class Decoded:
    opcode: int
    rd: int
    rs1: int
    rs2: int
    imm18: int  # sign-extended
    imm22: int  # sign-extended

    @property
    def group(self) -> int:
        return self.opcode >> 4 & 3


def decode(word: int) -> Decoded:
    return Decoded(
        opcode=word >> OP_SHIFT & 0x3F,
        rd=word >> RD_SHIFT & 0xF,
        rs1=word >> RS1_SHIFT & 0xF,
        rs2=word >> RS2_SHIFT & 0xF,
        imm18=sext(word, IMM18_BITS),
        imm22=sext(word, IMM22_BITS),
    )


SR_NAMES = {sr: "%" + sr.name.lower() for sr in SR}


def disassemble(word: int, symbols: dict[int, str] | None = None) -> str:
    d = decode(word)
    spec = BY_OPCODE.get(d.opcode)
    if spec is None:
        return f".word 0x{word:08x}"
    if word == 0:
        return "nop"

    def target(t: int) -> str:
        t &= (1 << IMM18_BITS) - 1
        return symbols.get(t, str(t)) if symbols else str(t)

    def mem(rs1: int, imm: int) -> str:
        if imm == 0:
            return f"[r{rs1}]"
        return f"[r{rs1} {'-' if imm < 0 else '+'} {abs(imm)}]"

    m, f = spec.mnemonic, spec.fmt
    if f == "rrr":
        return f"{m} r{d.rd}, r{d.rs1}, r{d.rs2}"
    if f == "rri":
        return f"{m} r{d.rd}, r{d.rs1}, {d.imm18}"
    if f == "ld":
        return f"{m} r{d.rd}, {mem(d.rs1, d.imm18)}"
    if f == "st":
        return f"{m} {mem(d.rs1, d.imm18)}, r{d.rd}"
    if f == "ri":
        return f"{m} r{d.rd}, {d.imm22}"
    if f == "sr":
        name = SR_NAMES.get(d.imm18 & 0x1F, str(d.imm18))
        return f"{m} r{d.rd}, {name}"
    if f == "br":
        return f"{m} r{d.rs1}, {target(d.imm18)}"
    if f == "tgt":
        return f"{m} {target(d.imm18)}"
    return m


# ------------------------------------------------------ Verilog header


def verilog_header() -> str:
    """Contents of rtl/rb_isa.vh (regenerate with `python -m rosenblatt.isa`)."""
    out = [
        "// rb_isa.vh: GENERATED by sw/rosenblatt/isa.py. Do not edit by hand.",
        "`ifndef RB_ISA_VH",
        "`define RB_ISA_VH",
        "",
    ]
    for fn in Fn:
        out.append(f"`define RB_FN_{fn.name:<6} 4'd{int(fn)}")
    out.append("")
    for op in Op:
        out.append(f"`define RB_OP_{op.name:<6} 6'h{int(op):02X}")
    out.append("")
    for sr in SR:
        out.append(f"`define RB_SR_{sr.name:<6} 5'd{int(sr)}")
    out.append("")
    for c in CSR:
        out.append(f"`define RB_CSR_{c.name:<9} 8'h{int(c):02X}")
    out.append(f"`define RB_PERF_STRIDE 8'h{PERF_STRIDE:02X}")
    out.append("")
    for p in Perf:
        out.append(f"`define RB_PERF_{p.name:<11} 4'd{int(p)}")
    out.append(f"`define RB_NUM_PERF {len(Perf)}")
    out.append("")
    out.append(f"`define RB_HOST_IMEM_BASE 24'h{HOST_IMEM_BASE:06X}")
    out.append(f"`define RB_HOST_CSR_BASE  24'h{HOST_CSR_BASE:06X}")
    out.append(f"`define RB_FEATURE_DP4A   8'h{FEATURE_DP4A:02X}")
    out.append(f"`define RB_HW_VERSION     8'd{HW_VERSION}")
    out += ["", "`endif", ""]
    return "\n".join(out)


if __name__ == "__main__":
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[2] / "rtl" / "rb_isa.vh"
    path.parent.mkdir(exist_ok=True)
    path.write_text(verilog_header(), newline="\n")
    print(f"wrote {path}")
