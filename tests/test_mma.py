"""mma, mmaz, mmar: a warp-wide int8 matrix multiply into an accumulator tile."""

import numpy as np
import pytest

from rosenblatt import isa
from rosenblatt.asm import assemble
from rosenblatt.device import Device, EmuBackend
from rosenblatt.emu import Config, EmuError

MMA = Config(mma=True)

# Each thread loads two words: a (r2, its row of A) and b (r3, its column of B).
# After BODY it stores the W rows of its tile column, then r2 and r3.
KERNEL = """
        s2r   r1, %gtid
        shli  r1, r1, 3
        s2r   r5, %arg0
        add   r5, r5, r1
        ldw   r2, [r5]
        ldw   r3, [r5 + 4]
        BODY
        s2r   r1, %gtid
        muli  r1, r1, 4 * (W + 2)
        s2r   r5, %arg1
        add   r5, r5, r1
.rept   W, j
        mmar  r6, j
        stw   [r5 + 4 * j], r6
.endr
        stw   [r5 + 4 * W], r2
        stw   [r5 + 4 * W + 4], r3
        ret
"""


def run(body: str, cfg: Config = MMA, warps: int = 1, seed: int = 0, full: bool = False):
    """Returns the input words (threads x 2, uint32), each thread's tile column
    (threads x W, int32; with full=True followed by r2 and r3) and the launch result."""
    rng = np.random.default_rng(seed)
    W = cfg.num_lanes
    n = W * warps
    words = rng.integers(0, 2**32, (n, 2), dtype=np.uint64).astype(np.uint32)
    dev = Device(EmuBackend(cfg))
    src, dst = dev.upload(words), dev.alloc(4 * (W + 2) * n)
    dev.load(assemble(KERNEL.replace("BODY", body), defines={"W": W}))
    result = dev.launch(grid=1, block=n, args=[src, dst])
    out = dev.download(dst, (W + 2) * n, np.int32).reshape(n, W + 2)
    return words, (out if full else out[:, :W]), result


def tile(a_words, b_words) -> np.ndarray:
    """C = A x B for one warp: row j of A is lane j's word, column i of B lane i's."""
    A = a_words.astype("<u4").view(np.int8).reshape(-1, 4).astype(np.int64)
    B = b_words.astype("<u4").view(np.int8).reshape(-1, 4).astype(np.int64)
    return A @ B.T          # C[j][i] = sum_k A[j][k] * B[i][k]


def test_one_mma_is_a_matrix_product():
    words, cols, _ = run("mmaz\n        mma r2, r3")
    C = tile(words[:, 0], words[:, 1])
    np.testing.assert_array_equal(cols.T, C)   # lane i holds column i


def test_mma_accumulates_and_mmaz_clears():
    body = """
        mmaz
        mma   r2, r3
        mma   r3, r2          ; rows from b, columns from a
        mma   r2, r3
        """
    words, cols, _ = run(body)
    a, b = words[:, 0], words[:, 1]
    np.testing.assert_array_equal(cols.T, 2 * tile(a, b) + tile(b, a))
    words, cols, _ = run(body + "mmaz\n        mma r3, r3")
    np.testing.assert_array_equal(cols.T, tile(words[:, 1], words[:, 1]))


def test_inactive_lanes_keep_their_column_but_still_supply_rows():
    body = """
        mmaz
        mma   r2, r3
        s2r   r4, %laneid
        andi  r4, r4, 1
        if.nz r4              ; odd lanes only
            mma   r3, r2
        endif
        """
    words, cols, _ = run(body)
    a, b = words[:, 0], words[:, 1]
    expect = tile(a, b) + np.where(np.arange(8) % 2 == 1, tile(b, a), 0)   # columns of odd lanes
    np.testing.assert_array_equal(cols.T, expect)


def test_mmaz_clears_only_the_active_lanes():
    body = """
        mmaz
        mma   r2, r3
        s2r   r4, %laneid
        slti  r4, r4, 3
        if.nz r4              ; lanes 0..2
            mmaz
        endif
        """
    words, cols, _ = run(body)
    expect = tile(words[:, 0], words[:, 1])
    expect[:, :3] = 0
    np.testing.assert_array_equal(cols.T, expect)


def test_every_warp_has_its_own_tile():
    words, cols, _ = run("mmaz\n        mma r2, r3", warps=4)
    for w in range(4):
        sl = slice(8 * w, 8 * w + 8)
        np.testing.assert_array_equal(cols[sl].T, tile(words[sl, 0], words[sl, 1]))


def test_four_lane_warps_have_four_by_four_tiles():
    cfg = Config(num_cores=1, num_lanes=4, num_warps=4, mma=True)
    words, cols, _ = run("mmaz\n        mma r2, r3", cfg, warps=2)
    for w in range(2):
        sl = slice(4 * w, 4 * w + 4)
        np.testing.assert_array_equal(cols[sl].T, tile(words[sl, 0], words[sl, 1]))


def test_mmar_row_wraps_at_the_warp_width():
    cfg = Config(num_cores=1, num_lanes=4, num_warps=2, mma=True)
    words, out, _ = run("mmaz\n        mma r2, r3\n        mmar r2, 6", cfg, full=True)
    C = tile(words[:, 0], words[:, 1])
    np.testing.assert_array_equal(out[:, 4], C[2])          # r2 = row 6 mod 4


def test_mma_reads_its_operands_before_the_tile_changes():
    # mmar overwrites r2 with row 1; a following mma must use the new r2
    words, cols, _ = run("mmaz\n        mma r2, r3\n        mmar r2, 1\n        mma r2, r3")
    a, b = words[:, 0], words[:, 1]
    C = tile(a, b)
    np.testing.assert_array_equal(cols.T, C + tile(C[1].astype(np.uint32), b))


def test_steps_are_counted():
    _, _, result = run("mmaz\n        mma r2, r3\n        mma r2, r3\n        mmar r4, 0", warps=2)
    assert result.stats["mma_steps"] == 2 * 3 * 8   # mma and mmaz: one step per tile row


@pytest.mark.parametrize("cfg", [Config(), Config(mma=True, dp4a=False)], ids=["no_mma", "no_dp4a"])
@pytest.mark.parametrize("body", ["mma r2, r3", "mmaz", "mmar r4, 0"])
def test_needs_the_matrix_unit(cfg, body):
    with pytest.raises(EmuError, match="needs the matrix unit"):
        run(body, cfg)


def test_encoding_roundtrip():
    for src, word in (("mma r2, r3", isa.encode(isa.Op.MMA, 0, 2, 3)),
                      ("mmaz", isa.encode(isa.Op.MMAZ)),
                      ("mmar r9, 7", isa.encode(isa.Op.MMAR, 9, lane=7))):
        assert assemble(src).words == [word]
        assert isa.disassemble(word) == src
    with pytest.raises(Exception, match="lane 16"):
        assemble("mmar r1, 16")
