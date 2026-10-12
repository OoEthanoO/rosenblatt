"""Test scenarios that run unchanged on the emulator and on the RTL.

Each scenario takes a Device and a seed, launches one or more kernels, checks
the results against an independent Python/NumPy reference where one exists,
and returns an Outcome (output arrays plus launch statistics). The RTL harness
runs every scenario on both the emulator and the simulated hardware and also
requires the two to agree exactly, including the event counters (instructions,
active lanes, coalesced memory requests, bank wavefronts, divergent branches).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rosenblatt.asm import assemble, assemble_file
from rosenblatt.device import Device, LaunchResult
from rosenblatt.isa import Fn, MASK32

KERNELS = Path(__file__).resolve().parents[1] / "kernels"


@dataclass
class Outcome:
    outputs: dict[str, np.ndarray] = field(default_factory=dict)
    runs: list[LaunchResult] = field(default_factory=list)


def zeros(dev: Device, nbytes: int) -> int:
    addr = dev.alloc(nbytes)
    dev.write(addr, bytes(nbytes))
    return addr


def tile_defines(dev: Device) -> dict[str, int]:
    W, NW = dev.cfg.num_lanes, dev.cfg.num_warps
    return {"LANES": W, "LOG_LANES": W.bit_length() - 1, "WARPS": NW, "LOG_WARPS": NW.bit_length() - 1}


# ------------------------------------------------------------------ basics


def vecadd(dev: Device, seed: int) -> Outcome:
    rng = np.random.default_rng(seed)
    n = 100
    a = rng.integers(-2**31, 2**31, n, dtype=np.int64).astype(np.int32)
    b = rng.integers(-2**31, 2**31, n, dtype=np.int64).astype(np.int32)
    pa, pb, pc = dev.upload(a), dev.upload(b), zeros(dev, n * 4)
    dev.load(assemble_file(str(KERNELS / "vecadd.s")))
    block = dev.cfg.max_threads
    run = dev.launch(grid=(n + block - 1) // block, block=block, args=[pa, pb, pc, n])
    c = dev.download(pc, n)
    assert (c == (a.astype(np.int64) + b).astype(np.int32)).all()
    return Outcome({"c": c}, [run])


def matmul_naive(dev: Device, seed: int) -> Outcome:
    rng = np.random.default_rng(seed)
    M, K, N = 6, 10, 8
    A = rng.integers(-1000, 1000, (M, K), dtype=np.int32)
    B = rng.integers(-1000, 1000, (K, N), dtype=np.int32)
    pa, pb, pc = dev.upload(A), dev.upload(B), zeros(dev, M * N * 4)
    dev.load(assemble_file(str(KERNELS / "matmul_naive.s")))
    block = dev.cfg.max_threads
    run = dev.launch(grid=(M * N + block - 1) // block, block=block,
                     args=[pa, pb, pc, M * N, K, N.bit_length() - 1])
    C = dev.download(pc, M * N).reshape(M, N)
    assert (C == A @ B).all()
    return Outcome({"C": C}, [run])


def special_regs(dev: Device, seed: int) -> Outcome:
    """Every thread stores its special registers; block size not a multiple of the warp."""
    names = ["tid", "ntid", "ctaid", "nctaid", "laneid", "warpid", "gtid", "arg0", "arg3", "arg7"]
    block, grid = dev.cfg.max_threads - 3, 3
    n = block * grid
    out = zeros(dev, n * len(names) * 4)
    lines = ["s2r r1, %gtid", f"muli r1, r1, {len(names) * 4}", "s2r r2, %arg1", "add r1, r1, r2"]
    for i, nm in enumerate(names):
        lines += [f"s2r r3, %{nm}", f"stw [r1 + {4 * i}], r3"]
    lines.append("ret")
    dev.load(assemble("\n".join(lines)))
    args = [0x1111, out, 0, 0xDEAD_BEEF, 0, 0, 0, 0x7777_7777]
    run = dev.launch(grid=grid, block=block, args=args)
    got = dev.download(out, n * len(names), np.uint32).reshape(n, len(names))
    W = dev.cfg.num_lanes
    for g in range(n):
        cta, tid = divmod(g, block)
        expect = [tid, block, cta, grid, tid % W, tid // W, g, args[0], args[3], args[7]]
        assert got[g].tolist() == expect, (g, got[g].tolist(), expect)
    return Outcome({"sr": got}, [run])


# --------------------------------------------------------- random ALU stress


def alu_random(dev: Device, seed: int, n_ops: int = 200) -> Outcome:
    """Random straight-line ALU code over random register contents.

    r1 points at the thread's 64-byte slot in the input buffer and is never a
    destination; r2..r15 are loaded from the slot, scrambled, and stored to the
    matching slot of the output half of the buffer.
    """
    rng = np.random.default_rng(seed)
    cfg = dev.cfg
    nthreads = cfg.num_lanes * 2
    special = [0, 1, 2, 0xFFFF_FFFF, 0x7FFF_FFFF, 0x8000_0000, 0xFF, 0x80, 0x7F,
               31, 32, 0x8000, 0xFFFF, 0xFFFF_8000]
    init = rng.integers(0, 2**32, (nthreads, 16), dtype=np.uint64)
    pick = rng.random((nthreads, 16)) < 0.3
    init[pick] = rng.choice(special, int(pick.sum()))
    buf = dev.alloc(2 * nthreads * 64)
    dev.write(buf, init.astype(np.uint32))
    dev.write(buf + nthreads * 64, bytes(nthreads * 64))
    out_off = nthreads * 64

    src = ["s2r r1, %gtid", "shli r1, r1, 6", "s2r r2, %arg0", "add r1, r1, r2"]
    src += [f"ldw r{r}, [r1 + {4 * r}]" for r in range(2, 16)]
    reg_fns = [f.name.lower() for f in Fn if f != Fn.DP4A or cfg.dp4a]
    reg_fns += ["dp4ab"] if cfg.dp4a else []
    imm_fns = [f for f in Fn if f != Fn.DP4A]
    sregs = ["%tid", "%ntid", "%ctaid", "%nctaid", "%laneid", "%warpid", "%gtid", "%arg0", "%arg1"]
    for _ in range(n_ops):
        rd = int(rng.integers(2, 16))
        rs1, rs2 = (int(x) for x in rng.integers(0, 16, 2))
        kind = rng.random()
        if kind < 0.55:
            fn = reg_fns[int(rng.integers(len(reg_fns)))]
            lane = f", {int(rng.integers(0, 16))}" if fn == "dp4ab" else ""
            src.append(f"{fn} r{rd}, r{rs1}, r{rs2}{lane}")
        elif kind < 0.9:
            fn = imm_fns[int(rng.integers(len(imm_fns)))]
            imm = int(rng.choice([0, 1, -1, 31, 32, 33, 131071, -131072,
                                  int(rng.integers(-131072, 131072)), int(rng.integers(-64, 64))]))
            src.append(f"{fn.name.lower()}i r{rd}, r{rs1}, {imm}")
        elif kind < 0.95:
            op = "li" if rng.random() < 0.5 else "lui"
            src.append(f"{op} r{rd}, {int(rng.integers(-(1 << 21), 1 << 21))}")
        else:
            src.append(f"s2r r{rd}, {sregs[int(rng.integers(len(sregs)))]}")
    src += [f"stw [r1 + {out_off + 4 * r}], r{r}" for r in range(2, 16)]
    src.append("ret")
    dev.load(assemble("\n".join(src)))
    run = dev.launch(grid=1, block=nthreads, args=[buf, 12345])
    out = dev.download(buf + out_off, nthreads * 16, np.uint32).reshape(nthreads, 16)
    return Outcome({"regs": out[:, 2:]}, [run])


def load_use_random(dev: Device, seed: int, n_ops: int = 160) -> Outcome:
    """Random mixes of global loads and the instructions that use them.

    Loaded registers are used at random distances: as ALU operands (including
    the accumulator of mad/dp4a), as store data, as the address of another load
    (pointer chasing), as branch conditions, and overwritten before or after
    their data arrives. Lanes read different words, so branches diverge. With
    the register scoreboard this is its stress test; with blocking loads it runs
    the same program the slow way. Some loads are still in flight at `bar` and
    at `ret`.
    """
    rng = np.random.default_rng(seed)
    cfg = dev.cfg
    nthreads = cfg.num_lanes * cfg.num_warps            # one full block
    data = dev.upload(rng.integers(0, 2**32, 64, dtype=np.uint64).astype(np.uint32))
    pointers = dev.upload((data + 4 * rng.integers(0, 64, 64)).astype(np.uint32))
    out = zeros(dev, 2 * nthreads * 128)                # 32 words per thread, 2 blocks

    # r1: this lane's window into the data, r13: into the pointers, r14: its
    # output slot. Random code reads and writes r2..r12 only.
    src = ["s2r r2, %gtid", "andi r3, r2, 15", "shli r3, r3, 2",
           "s2r r1, %arg0", "add r1, r1, r3", "s2r r13, %arg1", "add r13, r13, r3",
           "shli r2, r2, 7", "s2r r14, %arg2", "add r14, r14, r2"]
    src += [f"ldw r{r}, [r1 + {4 * r}]" for r in range(2, 13)]
    fns = ["add", "sub", "xor", "and", "or", "mul", "mad", "shl", "sra", "slt", "min"]
    fns += ["dp4a", "dp4ab"] if cfg.dp4a else []

    def reg() -> int:
        return int(rng.integers(2, 13))

    def alu(rd: int) -> str:
        fn = fns[int(rng.integers(len(fns)))]
        lane = f", {int(rng.integers(0, 16))}" if fn == "dp4ab" else ""
        return f"{fn} r{rd}, r{reg()}, r{reg()}{lane}"

    def simple(out_lines: list[str], allow_load: bool = True) -> None:
        kind = rng.random()
        if kind < 0.4 and allow_load:
            if rng.random() < 0.85:
                out_lines.append(f"ldw r{reg()}, [r1 + {4 * int(rng.integers(0, 48))}]")
            else:
                op = "ldb" if rng.random() < 0.5 else "ldbu"
                out_lines.append(f"{op} r{reg()}, [r1 + {int(rng.integers(0, 192))}]")
        elif kind < 0.85:
            out_lines.append(alu(reg()))
        else:
            out_lines.append(f"stw [r14 + {4 * int(rng.integers(0, 16))}], r{reg()}")

    for _ in range(n_ops):
        kind = rng.random()
        if kind < 0.6:
            simple(src)
        elif kind < 0.72:                               # pointer chase, maybe with a gap
            ptr, dst = reg(), reg()
            src.append(f"ldw r{ptr}, [r13 + {4 * int(rng.integers(0, 48))}]")
            for _ in range(int(rng.integers(0, 3))):
                tgt = reg()
                while tgt == ptr:
                    tgt = reg()
                src.append(alu(tgt))
            src.append(f"ldw r{dst}, [r{ptr}]")
        elif kind < 0.8:                                # same register loaded twice in a row
            r = reg()
            src += [f"ldw r{r}, [r1 + {4 * int(rng.integers(0, 48))}]",
                    f"ldw r{r}, [r1 + {4 * int(rng.integers(0, 48))}]"]
        elif kind < 0.95:                               # branch on a (maybe still loading) value
            src.append(f"if.nz r{reg()}")
            for _ in range(int(rng.integers(1, 4))):
                simple(src)
            if rng.random() < 0.4:
                src.append("else")
                simple(src)
            src.append("endif")
        else:
            src.append("bar")
    src += [f"stw [r14 + {64 + 4 * (r - 2)}], r{r}" for r in range(2, 13)]
    src += ["ldw r12, [r1]", "ret"]                     # still in flight at ret
    dev.load(assemble("\n".join(src)))
    run = dev.launch(grid=2, block=nthreads, args=[data, pointers, out])
    words = dev.download(out, 2 * nthreads * 32, np.uint32).reshape(2 * nthreads, 32)
    return Outcome({"words": words}, [run])


# ---------------------------------------------------- random divergence stress


def _gen_block(rng, depth: int, max_depth: int, n: int) -> list:
    stmts = []
    for _ in range(n):
        r = rng.random()
        if depth < max_depth and r < 0.30:
            then = _gen_block(rng, depth + 1, max_depth, int(rng.integers(1, 4)))
            other = _gen_block(rng, depth + 1, max_depth, int(rng.integers(1, 3))) if rng.random() < 0.6 else None
            stmts.append(("if", int(rng.integers(0, 30)), rng.random() < 0.5, then, other))
        elif depth < max_depth and r < 0.45:
            body = _gen_block(rng, depth + 1, max_depth, int(rng.integers(1, 3)))
            stmts.append(("loop", int(rng.integers(0, 29)), depth, body))
        elif depth < max_depth and r < 0.52:
            stmts.append(("ret", int(rng.integers(0, 30))))
        else:
            stmts.append(("op", int(rng.integers(0, 3)), int(rng.integers(-1000, 1000))))
    return stmts


def _emit(stmts: list, out: list[str]) -> None:
    for s in stmts:
        if s[0] == "op":
            _, kind, c = s
            if kind == 0:
                out += ["muli r3, r3, 3", f"addi r3, r3, {c}"]
            elif kind == 1:
                out.append(f"xori r3, r3, {c}")
            else:
                out += [f"addi r3, r3, {c}", "shli r4, r3, 7", "xor r3, r3, r4"]
        elif s[0] == "if":
            _, bit, nz, then, other = s
            out += [f"shri r4, r2, {bit}", "andi r4, r4, 1", f"if.{'nz' if nz else 'z'} r4"]
            _emit(then, out)
            if other is not None:
                out.append("else")
                _emit(other, out)
            out.append("endif")
        elif s[0] == "loop":
            _, bit, depth, body = s
            c = f"r{5 + depth}"
            out += [f"shri {c}, r2, {bit}", f"andi {c}, {c}, 3", f"addi {c}, {c}, 1", "loop"]
            _emit(body, out)
            out += [f"subi {c}, {c}, 1", f"endloop.nz {c}"]
        else:  # early return of the threads whose bit is set
            _, bit = s
            out += [f"shri r4, r2, {bit}", "andi r4, r4, 1", "if.nz r4",
                    "stw [r1 + 4], r3", "ret", "endif"]


def _interp(stmts: list, x: int, acc: int, mul16: bool = False) -> tuple[int, bool]:
    """Sequential (one thread at a time) meaning of the generated program."""
    for s in stmts:
        if s[0] == "op":
            _, kind, c = s
            if kind == 0:
                a = (acc & 0xFFFF) - ((acc & 0x8000) << 1) if mul16 else acc
                acc = (a * 3 + c) & MASK32
            elif kind == 1:
                acc = (acc ^ (c & MASK32)) & MASK32
            else:
                acc = (acc + c) & MASK32
                acc ^= (acc << 7) & MASK32
        elif s[0] == "if":
            _, bit, nz, then, other = s
            taken = (x >> bit & 1) == (1 if nz else 0)
            branch = then if taken else (other or [])
            acc, done = _interp(branch, x, acc, mul16)
            if done:
                return acc, True
        elif s[0] == "loop":
            _, bit, _depth, body = s
            for _ in range((x >> bit & 3) + 1):
                acc, done = _interp(body, x, acc, mul16)
                if done:
                    return acc, True
        else:
            if x >> s[1] & 1:
                return acc, True
    return acc, False


def divergence_random(dev: Device, seed: int) -> Outcome:
    """Random nested if/else, data-dependent loops and early returns.

    The SIMT hardware runs warps in lockstep with masks and a reconvergence
    stack; the result must equal running every thread on its own.
    """
    rng = np.random.default_rng(seed)
    cfg = dev.cfg
    prog = _gen_block(rng, 0, 3, 6)
    src = ["s2r r1, %gtid", "shli r1, r1, 3", "s2r r2, %arg0", "add r1, r1, r2",
           "ldw r2, [r1]", "li r3, 0"]
    _emit(prog, src)
    src += ["stw [r1 + 4], r3", "ret"]
    block = cfg.max_threads
    grid = 2
    n = block * grid
    data = rng.integers(0, 2**32, n, dtype=np.uint64)
    # make some branches uniform per warp: bits 20..29 equal within a warp
    for w in range(n // cfg.num_lanes):
        hi = int(rng.integers(0, 1 << 10)) << 20
        sl = slice(w * cfg.num_lanes, (w + 1) * cfg.num_lanes)
        data[sl] = (data[sl] & ~np.uint64(0x3FF0_0000)) | np.uint64(hi)
    buf = dev.alloc(n * 8)
    words = np.zeros((n, 2), dtype=np.uint32)
    words[:, 0] = data.astype(np.uint32)
    dev.write(buf, words)
    dev.load(assemble("\n".join(src)))
    run = dev.launch(grid=grid, block=block, args=[buf])
    got = dev.download(buf, n * 2, np.uint32).reshape(n, 2)[:, 1]
    mul16 = cfg.mul_width == 16
    expect = np.array([_interp(prog, int(x), 0, mul16)[0] for x in data], dtype=np.uint32)
    bad = np.nonzero(got != expect)[0]
    assert bad.size == 0, f"threads {bad[:8].tolist()} differ: got {got[bad[:4]]} expect {expect[bad[:4]]}"
    return Outcome({"acc": got}, [run])


# ------------------------------------------------------------ memory system

_MEM_KERNEL = """
; Exercises every load/store flavor and the coalescer / bank logic.
; args: arg0 src table (1 KiB), arg1 dst (64 bytes per thread), arg2 index table,
;       arg3 collision area (16 bytes per warp)
.alias out, r12
        s2r   r1, %gtid
        s2r   r2, %tid
        s2r   r10, %arg0
        s2r   r11, %arg1
        shli  out, r1, 6
        add   out, out, r11
        ; 1. coalesced words
        shli  r3, r1, 2
        add   r3, r3, r10
        ldw   r4, [r3]
        stw   [out + 0], r4
        ; 2-3. consecutive bytes, signed and unsigned (4 lanes per word)
        add   r3, r10, r1
        ldb   r4, [r3 + 3]
        stw   [out + 4], r4
        ldbu  r4, [r3 + 5]
        stw   [out + 8], r4
        ; 4. broadcast
        ldw   r4, [r10 + 12]
        stw   [out + 12], r4
        ; 5. strided words
        muli  r3, r1, 36
        andi  r3, r3, 1020
        add   r3, r3, r10
        ldw   r4, [r3]
        stw   [out + 16], r4
        ; 6. gather through an index table (bytes and words)
        s2r   r5, %arg2
        shli  r3, r1, 2
        add   r3, r3, r5
        ldw   r6, [r3]
        add   r6, r6, r10
        ldb   r4, [r6]
        stw   [out + 20], r4
        andi  r6, r6, -4
        ldw   r4, [r6]
        stw   [out + 24], r4
        ; 7. scattered byte stores (4 lanes merge into one word)
        stb   [out + 28], r1
        addi  r4, r1, 1
        stb   [out + 30], r4
        ; 8. collisions inside one warp: the highest active lane wins
        s2r   r5, %arg3
        s2r   r6, %warpid
        s2r   r7, %ctaid
        shli  r7, r7, LOG_WARPS
        add   r6, r6, r7
        shli  r6, r6, 4
        add   r5, r5, r6            ; 16-byte area for this warp
        stw   [r5], r1              ; every lane writes the same word
        s2r   r7, %laneid
        andi  r8, r7, 3
        add   r8, r8, r5
        stb   [r8 + 4], r7          ; lanes l and l+4 hit the same byte
        ; 9. shared memory: neighbour exchange through a barrier
        shli  r3, r2, 2
        sts   [r3], r1
        bar
        xori  r4, r2, 1
        shli  r4, r4, 2
        lds   r5, [r4]
        stw   [out + 32], r5
        bar
        ; 10. worst-case bank conflicts: stride of NUM_LANES words
        shli  r3, r2, SHIFT_CONFLICT
        addi  r4, r1, 100
        sts   [r3 + 4], r4
        lds   r5, [r3 + 4]
        stw   [out + 36], r5
        bar
        ; 11. shared byte stores / loads, broadcast read
        stsb  [r2 + 512], r2
        bar
        xori  r4, r2, 3
        ldsbu r5, [r4 + 512]
        stw   [out + 40], r5
        ldsb  r5, [r0 + 515]
        stw   [out + 44], r5
        lds   r5, [r0 + 512]
        stw   [out + 48], r5
        ret
"""


def memory_patterns(dev: Device, seed: int) -> Outcome:
    rng = np.random.default_rng(seed)
    cfg = dev.cfg
    block, grid = cfg.max_threads, 2
    n = block * grid
    src = rng.integers(0, 256, 1024, dtype=np.int64).astype(np.uint8)
    idx = rng.integers(0, 1024, n, dtype=np.int64).astype(np.uint32)
    psrc, pidx = dev.upload(src), dev.upload(idx)
    pdst = zeros(dev, n * 64)
    pcol = zeros(dev, grid * cfg.num_warps * 16)
    shift = 2 + (cfg.num_lanes - 1).bit_length()
    dev.load(assemble(_MEM_KERNEL, defines={"SHIFT_CONFLICT": shift, **tile_defines(dev)}))
    run = dev.launch(grid=grid, block=block, args=[psrc, pdst, pidx, pcol])
    dst = dev.download(pdst, n * 16, np.uint32).reshape(n, 16)
    col = dev.download(pcol, grid * cfg.num_warps * 4, np.uint32).reshape(-1, 4)

    w = src.view(np.uint8)
    word = lambda a: int.from_bytes(bytes(w[a & ~3: (a & ~3) + 4]), "little")  # noqa: E731
    sb = lambda v: (v - 256 if v >= 128 else v) & MASK32  # noqa: E731
    for g in range(n):
        tid = g % block
        assert dst[g, 0] == word(4 * g)
        assert dst[g, 1] == sb(int(w[g + 3]))
        assert dst[g, 2] == int(w[g + 5])
        assert dst[g, 3] == word(12)
        assert dst[g, 4] == word((36 * g) & 1020)
        assert dst[g, 5] == sb(int(w[idx[g]]))
        assert dst[g, 6] == word(int(idx[g]))
        assert dst[g, 7] == (g & 0xFF) | ((g + 1) & 0xFF) << 16
        assert dst[g, 8] == g - tid + (tid ^ 1)
        assert dst[g, 9] == g + 100
        assert dst[g, 10] == tid ^ 3
        assert dst[g, 11] == 3
        assert dst[g, 12] == 0x03020100
    for slot in range(grid * cfg.num_warps):
        cta, wid = divmod(slot, cfg.num_warps)
        last = cta * block + wid * cfg.num_lanes + cfg.num_lanes - 1
        assert col[slot, 0] == last
        lanes = cfg.num_lanes
        expect = bytes(max(l for l in range(lanes) if l % 4 == b) for b in range(4))
        assert col[slot, 1] == int.from_bytes(expect, "little")
    return Outcome({"dst": dst, "col": col}, [run])


# ------------------------------------------------------ barriers / reduction


def reduce_sum(dev: Device, seed: int) -> Outcome:
    rng = np.random.default_rng(seed)
    block, grid = dev.cfg.max_threads, 3
    x = rng.integers(-10**6, 10**6, block * grid, dtype=np.int64).astype(np.int32)
    px, py = dev.upload(x), zeros(dev, grid * 4)
    dev.load(assemble_file(str(KERNELS / "reduce_sum.s")))
    run = dev.launch(grid=grid, block=block, args=[px, py])
    y = dev.download(py, grid)
    assert (y == x.reshape(grid, block).sum(axis=1)).all(), (y, x.reshape(grid, block).sum(axis=1))
    return Outcome({"y": y}, [run])


def matmul_tiled(dev: Device, seed: int) -> Outcome:
    rng = np.random.default_rng(seed)
    M, K, N = 8, 16, 16
    A = rng.integers(-1000, 1000, (M, K), dtype=np.int32)
    B = rng.integers(-1000, 1000, (K, N), dtype=np.int32)
    pa, pb, pc = dev.upload(A), dev.upload(B), zeros(dev, M * N * 4)
    dev.load(assemble_file(str(KERNELS / "matmul_tiled.s"), defines=tile_defines(dev)))
    W, rows = dev.cfg.num_lanes, dev.cfg.num_warps   # tile = num_warps x num_lanes outputs
    grid = (M // rows) * (N // W)
    run = dev.launch(grid=grid, block=rows * W, args=[pa, pb, pc, K, N, (N // W).bit_length() - 1])
    C = dev.download(pc, M * N).reshape(M, N)
    assert (C == A @ B).all()
    return Outcome({"C": C}, [run])


def mlp_int8(dev: Device, seed: int) -> Outcome:
    """Two dense layers, padding, partial warps, and reusable device buffers;
    both weight layouts and the unrolled loop when the device has DP4A."""
    from ml.gpu import GpuMLP
    from ml.qmlp import QuantMLP

    rng = np.random.default_rng(seed)
    model = QuantMLP(
        rng.integers(-128, 128, (13, 7), dtype=np.int16).astype(np.int8),
        rng.integers(-10000, 10000, 13, dtype=np.int32), 7,
        rng.integers(-128, 128, (10, 13), dtype=np.int16).astype(np.int8),
        rng.integers(-1000, 1000, 10, dtype=np.int32),
    )
    gpus = {"": GpuMLP(dev, model)}
    if dev.cfg.dp4a:
        gpus["il_"] = GpuMLP(dev, model, layout="interleaved")
        gpus["u2_"] = GpuMLP(dev, model, unroll=2)
        gpus["il_u2_"] = GpuMLP(dev, model, layout="interleaved", unroll=2)
    outcome = Outcome()
    for i, q0 in enumerate((np.zeros(7, dtype=np.int32), rng.integers(0, 128, 7, dtype=np.int32))):
        for tag, gpu in gpus.items():
            result = gpu.run(q0)
            np.testing.assert_array_equal(result.hidden, model.hidden(q0))
            np.testing.assert_array_equal(result.logits, model.logits(q0))
            assert result.prediction == int(np.argmax(model.logits(q0)))
            outcome.outputs[f"{tag}hidden_{i}"] = result.hidden
            outcome.outputs[f"{tag}logits_{i}"] = result.logits
            outcome.runs.extend(result.runs)
    return outcome


def mlp_trained(dev: Device, seed: int) -> Outcome:
    """Full 784 -> 64 -> 10 trained model; deterministic input, no downloads."""
    from ml.gpu import GpuMLP
    from ml.qmlp import QuantMLP

    model = QuantMLP.load()
    q0 = np.random.default_rng(seed).integers(0, 128, model.w1.shape[1], dtype=np.int32)
    result = GpuMLP(dev, model).run(q0)
    np.testing.assert_array_equal(result.hidden, model.hidden(q0))
    np.testing.assert_array_equal(result.logits, model.logits(q0))
    return Outcome({"hidden": result.hidden, "logits": result.logits}, result.runs)


def mlp_tiled(dev: Device, seed: int) -> Outcome:
    """Batched, shared-memory tiled inference (BatchMLP) with padded odd sizes,
    with and without DP4A, both weight layouts, dp4ab broadcasts and
    double-buffered tiles.

    Only meaningful on devices with 32-thread blocks; elsewhere it does nothing.
    """
    from ml.gpu import BatchMLP
    from ml.qmlp import QuantMLP

    if dev.cfg.max_threads != BatchMLP.BT * BatchMLP.KC // 4:
        return Outcome()
    rng = np.random.default_rng(seed)
    model = QuantMLP(
        rng.integers(-128, 128, (40, 45), dtype=np.int16).astype(np.int8),
        rng.integers(-20000, 20000, 40, dtype=np.int32), 8,
        rng.integers(-128, 128, (10, 40), dtype=np.int16).astype(np.int8),
        rng.integers(-1000, 1000, 10, dtype=np.int32),
    )
    q0 = rng.integers(0, 128, (11, 45), dtype=np.int32)
    outcome = Outcome()
    for dp4a in ([True, False] if dev.cfg.dp4a else [False]):
        res = BatchMLP(dev, model, batch=11, dp4a=dp4a).run(q0)
        np.testing.assert_array_equal(res.hidden, model.hidden(q0))
        np.testing.assert_array_equal(res.logits, model.logits(q0))
        outcome.outputs[f"logits_{int(dp4a)}"] = res.logits
        outcome.runs.extend(res.runs)
    res = BatchMLP(dev, model, batch=11, dp4a=dev.cfg.dp4a, layout="interleaved").run(q0)
    np.testing.assert_array_equal(res.logits, model.logits(q0))
    outcome.outputs["logits_interleaved"] = res.logits
    outcome.runs.extend(res.runs)
    if dev.cfg.dp4a and dev.cfg.num_lanes == BatchMLP.BT:
        for dbuf in (False, True):
            res = BatchMLP(dev, model, batch=11, bcast=True, dbuf=dbuf).run(q0)
            np.testing.assert_array_equal(res.hidden, model.hidden(q0))
            np.testing.assert_array_equal(res.logits, model.logits(q0))
            outcome.outputs[f"logits_bcast_{int(dbuf)}"] = res.logits
            outcome.runs.extend(res.runs)
    return outcome


SCENARIOS = {
    "vecadd": vecadd,
    "matmul_naive": matmul_naive,
    "special_regs": special_regs,
    "alu_random": alu_random,
    "load_use_random": load_use_random,
    "divergence_random": divergence_random,
    "memory_patterns": memory_patterns,
    "reduce_sum": reduce_sum,
    "matmul_tiled": matmul_tiled,
    "mlp_int8": mlp_int8,
    "mlp_trained": mlp_trained,
    "mlp_tiled": mlp_tiled,
}
