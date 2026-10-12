"""dp4ab: dp4a with the first operand broadcast from one lane of the warp."""

import numpy as np
import pytest

from rosenblatt import isa
from rosenblatt.asm import assemble
from rosenblatt.device import Device, EmuBackend
from rosenblatt.emu import Config, EmuError

# each thread loads two words (a, b) of its own, then runs `body`, then stores r4 and r2
KERNEL = """
        s2r   r1, %gtid
        shli  r1, r1, 3
        s2r   r5, %arg0
        add   r5, r5, r1
        ldw   r2, [r5]
        ldw   r3, [r5 + 4]
        li    r4, 1000
        BODY
        s2r   r5, %arg1
        add   r5, r5, r1
        stw   [r5], r4
        stw   [r5 + 4], r2
        ret
"""


def dot4(a: int, b: int) -> int:
    s8 = lambda v, k: ((v >> k) & 0xFF) - (((v >> k) & 0x80) << 1)  # noqa: E731
    return sum(s8(a, k) * s8(b, k) for k in (0, 8, 16, 24))


def run(body: str, cfg: Config = Config(), seed: int = 0):
    rng = np.random.default_rng(seed)
    n = cfg.num_lanes
    words = rng.integers(0, 2**32, (n, 2), dtype=np.uint64).astype(np.uint32)
    dev = Device(EmuBackend(cfg))
    src, dst = dev.upload(words), dev.alloc(8 * n)
    dev.load(assemble(KERNEL.replace("BODY", body)))
    run = dev.launch(grid=1, block=n, args=[src, dst])
    return [int(w) for w in words[:, 0]], [int(w) for w in words[:, 1]], \
        dev.download(dst, 2 * n, np.uint32).reshape(n, 2), run


@pytest.mark.parametrize("lane", range(8))
def test_every_lane_reads_the_broadcast_operand(lane):
    a, b, out, _ = run(f"dp4ab r4, r2, r3, {lane}")
    for t in range(8):
        assert int(out[t, 0]) == (1000 + dot4(a[lane], b[t])) & isa.MASK32


def test_lane_index_wraps_at_the_warp_width():
    cfg = Config(num_cores=1, num_lanes=4, num_warps=2)
    a, b, out, _ = run("dp4ab r4, r2, r3, 6", cfg)
    for t in range(4):
        assert int(out[t, 0]) == (1000 + dot4(a[2], b[t])) & isa.MASK32


def test_all_lanes_read_before_any_lane_writes():
    # rd == rs1: every lane must use lane 1's old r2, not one already overwritten
    a, b, out, _ = run("mov r2, r2\n        dp4ab r2, r2, r3, 1")
    for t in range(8):
        assert int(out[t, 1]) == (a[t] + dot4(a[1], b[t])) & isa.MASK32


def test_needs_dp4a():
    with pytest.raises(EmuError, match="dp4ab needs dp4a"):
        run("dp4ab r4, r2, r3, 0", Config(dp4a=False))


def test_encoding_roundtrip():
    word = isa.encode(isa.Op.DP4AB, 8, 2, 1, lane=7)
    assert isa.disassemble(word) == "dp4ab r8, r2, r1, 7"
    assert assemble("dp4ab r8, r2, r1, 7").words == [word]
    with pytest.raises(ValueError):
        isa.encode(isa.Op.DP4AB, 8, 2, 1, lane=16)
