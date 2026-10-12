"""How global-memory requests are counted for each line size (the RTL tests
check that the hardware's counters agree with these emulator counts)."""

import numpy as np
import pytest

from rosenblatt.asm import assemble
from rosenblatt.device import Device, EmuBackend
from rosenblatt.emu import Config

# one warp copies one element per lane, STRIDE bytes apart, from %arg0 to %arg1
COPY = """
        s2r   r1, %tid
        muli  r1, r1, STRIDE
        s2r   r2, %arg0
        add   r2, r2, r1
        s2r   r3, %arg1
        add   r3, r3, r1
.if WORDS
        ldw   r4, [r2]
        stw   [r3], r4
.else
        ldbu  r4, [r2]
        stb   [r3], r4
.endif
        ret
"""


@pytest.mark.parametrize("line, words, stride, loads, stores", [
    (4, True, 4, 8, 8),       # consecutive words: one request per word
    (16, True, 4, 2, 8),      # loads share lines; stores coalesce only within a word
    (32, True, 4, 1, 8),
    (4, False, 1, 2, 2),      # consecutive bytes: four lanes share each word
    (32, False, 1, 1, 2),
    (32, True, 64, 8, 8),     # a row apart: every lane needs its own line
    (32, True, 0, 1, 1),      # broadcast
])
def test_request_counts(line, words, stride, loads, stores):
    dev = Device(EmuBackend(Config(gmem_line_bytes=line)))
    src = dev.upload(np.arange(1024, dtype=np.int32).view(np.uint8), align=32)
    dst = dev.alloc(4096, align=32)
    dev.load(assemble(COPY, defines={"WORDS": int(words), "STRIDE": stride}))
    run = dev.launch(grid=1, block=8, args=[src, dst])
    assert (run.stats["gmem_loads"], run.stats["gmem_stores"]) == (1, 1)
    assert run.stats["gmem_reqs"] == loads + stores
    size = 4 if words else 1
    data, out = dev.download(src, 4096, np.uint8), dev.download(dst, 4096, np.uint8)
    for lane in range(8):
        lo = lane * stride
        np.testing.assert_array_equal(out[lo:lo + size], data[lo:lo + size])
