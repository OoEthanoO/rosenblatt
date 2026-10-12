"""Fast inference checks that do not require an RTL simulator or downloads."""

import numpy as np
import pytest

from ml.gpu import LAYOUTS, BatchMLP, GpuMLP, interleave_words
from ml.qmlp import QuantMLP, preprocess
from rosenblatt.device import Device, EmuBackend
from rosenblatt.emu import Config
from scenarios import mlp_int8


@pytest.mark.parametrize("cfg", [Config(), Config(dp4a=False),
                                 Config(num_cores=1, num_lanes=4, num_warps=2,
                                        mul_width=16, dp4a=False)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_network(cfg, seed):
    mlp_int8(Device(EmuBackend(cfg)), seed)


def rounding_model():
    # At shift=2 these exercise negative/zero, half-way rounding and saturation.
    bias = np.array([-100, -3, -2, -1, 0, 1, 2, 3, 4, 505, 506, 507, 1000], dtype=np.int32)
    return QuantMLP(np.zeros((13, 3), dtype=np.int8), bias, 2,
                    np.zeros((5, 13), dtype=np.int8), np.array([-8, 7, 7, -1, -9], dtype=np.int32))


@pytest.mark.parametrize("dp4a", [True, False])
def test_rounding_saturation_and_argmax_ties(dp4a):
    model = rounding_model()
    result = GpuMLP(Device(EmuBackend()), model, dp4a=dp4a).run(np.zeros(3, dtype=np.int32))
    np.testing.assert_array_equal(result.hidden, [0, 0, 0, 0, 0, 0, 1, 1, 1, 126, 127, 127, 127])
    np.testing.assert_array_equal(result.logits, [-8, 7, 7, -1, -9])
    assert result.prediction == 1


def test_trained_network_matches_both_paths():
    model = QuantMLP.load()
    pixels = np.random.default_rng(17).integers(0, 256, (28, 28), dtype=np.uint8)
    q0 = preprocess(pixels)
    packed = GpuMLP(Device(EmuBackend()), model).run(q0)
    scalar = GpuMLP(Device(EmuBackend(Config(mul_width=16, dp4a=False))), model).run(q0)
    for result in (packed, scalar):
        np.testing.assert_array_equal(result.hidden, model.hidden(q0))
        np.testing.assert_array_equal(result.logits, model.logits(q0))
    assert sum(r.stats["instructions"] for r in packed.runs) < sum(r.stats["instructions"] for r in scalar.runs)
    assert sum(r.stats["gmem_reqs"] for r in packed.runs) < sum(r.stats["gmem_reqs"] for r in scalar.runs)


@pytest.mark.parametrize("bad_input", [np.zeros(2, dtype=np.int32), np.array([0, -1, 127]),
                                        np.array([0, 128, 127]), np.zeros(3, dtype=float)])
def test_reject_invalid_input(bad_input):
    gpu = GpuMLP(Device(EmuBackend()), rounding_model())
    with pytest.raises(ValueError, match="input"):
        gpu.run(bad_input)


@pytest.mark.parametrize("shift", [0, 32, 1.5])
def test_reject_invalid_shift(shift):
    model = rounding_model()
    model.sh1 = shift
    with pytest.raises(ValueError, match="shift"):
        GpuMLP(Device(EmuBackend()), model)


def test_reject_accumulator_overflow():
    model = rounding_model()
    model.b1[0] = (1 << 31) - 1
    with pytest.raises(ValueError, match="overflow"):
        GpuMLP(Device(EmuBackend()), model)


def test_reject_unsupported_dp4a():
    with pytest.raises(ValueError, match="DP4A"):
        GpuMLP(Device(EmuBackend(Config(dp4a=False))), rounding_model(), dp4a=True)


@pytest.mark.parametrize("dp4a", [True, False])
def test_batched_tiled_network_matches_reference(dp4a):
    model = QuantMLP.load()
    q0 = np.random.default_rng(5).integers(0, 128, (13, 784), dtype=np.int32)
    result = BatchMLP(Device(EmuBackend()), model, batch=13, dp4a=dp4a).run(q0)
    np.testing.assert_array_equal(result.hidden, model.hidden(q0))
    np.testing.assert_array_equal(result.logits, model.logits(q0))
    np.testing.assert_array_equal(result.predictions, np.argmax(model.logits(q0), axis=1))


def test_batched_kernel_cuts_global_memory_traffic():
    model = QuantMLP.load()
    q0 = np.random.default_rng(6).integers(0, 128, (16, 784), dtype=np.int32)
    tiled = BatchMLP(Device(EmuBackend()), model, batch=16).run(q0)
    per_image = GpuMLP(Device(EmuBackend()), model).run(q0[0])
    tiled_reqs = sum(r.stats["gmem_reqs"] for r in tiled.runs) / 16
    single_reqs = sum(r.stats["gmem_reqs"] for r in per_image.runs)
    assert tiled_reqs * 5 < single_reqs   # weights are reused across the 8 images of a block


@pytest.mark.parametrize("cfg", [Config(num_lanes=4), Config(num_warps=2)])
def test_batched_needs_32_thread_blocks(cfg):
    with pytest.raises(ValueError, match="32-thread"):
        BatchMLP(Device(EmuBackend(cfg)), rounding_model(), batch=8)


def test_interleave_words_layout():
    w = np.arange(24, dtype=np.int8).reshape(3, 8)
    flat = interleave_words(w).reshape(-1)
    for n in range(3):
        for k in range(8):
            assert flat[4 * ((k // 4) * 3 + n) + k % 4] == w[n, k]


def test_reject_bad_layouts():
    with pytest.raises(ValueError, match="DP4A"):
        GpuMLP(Device(EmuBackend()), rounding_model(), dp4a=False, layout="interleaved")
    with pytest.raises(ValueError, match="layout"):
        GpuMLP(Device(EmuBackend()), rounding_model(), layout="columns")
    with pytest.raises(ValueError, match="layout"):
        BatchMLP(Device(EmuBackend()), rounding_model(), batch=8, layout="columns")
    with pytest.raises(ValueError, match="unroll"):
        GpuMLP(Device(EmuBackend()), rounding_model(), unroll=3)
    with pytest.raises(ValueError, match="unroll"):
        GpuMLP(Device(EmuBackend()), rounding_model(), dp4a=False, unroll=2)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_unrolled_loop_matches_and_saves_instructions(layout):
    model = QuantMLP.load()
    q0 = np.random.default_rng(9).integers(0, 128, model.w1.shape[1], dtype=np.int32)
    plain, unrolled = (GpuMLP(Device(EmuBackend()), model, layout=layout, unroll=u).run(q0) for u in (1, 2))
    for result in (plain, unrolled):
        np.testing.assert_array_equal(result.hidden, model.hidden(q0))
        np.testing.assert_array_equal(result.logits, model.logits(q0))
    count = lambda r: sum(run.stats["instructions"] for run in r.runs)  # noqa: E731
    assert count(unrolled) < 0.85 * count(plain)


def test_wide_lines_pay_off_only_with_the_interleaved_layout():
    """With 32-byte memory lines, a warp's weight loads share requests only when
    neighbouring neurons' weights are neighbours in memory."""
    model = QuantMLP.load()
    q0 = np.random.default_rng(5).integers(0, 128, (8, model.w1.shape[1]), dtype=np.int32)
    reqs = {}
    for layout in LAYOUTS:
        single = GpuMLP(Device(EmuBackend(Config(gmem_line_bytes=32))), model, layout=layout).run(q0[0])
        np.testing.assert_array_equal(single.logits, model.logits(q0[0]))
        batch = BatchMLP(Device(EmuBackend(Config(gmem_line_bytes=32))), model, batch=8,
                         layout=layout).run(q0)
        np.testing.assert_array_equal(batch.logits, model.logits(q0))
        reqs[layout] = [sum(r.stats["gmem_reqs"] for r in res.runs) for res in (single, batch)]
    assert all(4 * il < rows for il, rows in zip(reqs["interleaved"], reqs["rows"]))


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dp4a", [True, False])
def test_stream_trained_network_through_64kib(layout, dp4a):
    model = QuantMLP.load()
    dev = Device(EmuBackend(Config(gmem_bytes=64 * 1024)))
    # Exercise alignment with a live allocation preceding the network.
    prefix = b"keep these bytes!"
    dev.write(dev.alloc(len(prefix)), prefix)
    before = dev.memory_used
    gpu = BatchMLP(dev, model, dp4a=dp4a, layout=layout)
    assert gpu.batch == 8
    assert gpu.memory_bytes == dev.memory_used - before
    assert dev.memory_used == 60_416 + 32
    q0 = np.random.default_rng(43).integers(0, 128, (19, 784), dtype=np.int32)
    chunks = list(gpu.run_batches(q0))
    assert [len(chunk.logits) for chunk in chunks] == [8, 8, 3]
    assert [chunk.padded_count for chunk in chunks] == [8, 8, 8]
    np.testing.assert_array_equal(np.concatenate([chunk.hidden for chunk in chunks]), model.hidden(q0))
    np.testing.assert_array_equal(np.concatenate([chunk.logits for chunk in chunks]), model.logits(q0))
    assert dev.read(0, len(prefix)) == prefix
    assert dev.memory_used == 60_416 + 32  # no allocations per inference


def test_partial_batch_does_not_execute_all_reserved_slots():
    model = rounding_model()
    q0 = np.zeros((9, 3), dtype=np.int32)
    dev = Device(EmuBackend())
    gpu = BatchMLP(dev, model, batch=32)
    first = gpu.run(np.full((32, 3), 127, dtype=np.int32))
    last = gpu.run(q0)
    exact = BatchMLP(Device(EmuBackend()), model, batch=16).run(q0)
    assert first.padded_count == 32 and last.padded_count == 16
    assert [run.stats for run in last.runs] == [run.stats for run in exact.runs]
    np.testing.assert_array_equal(last.hidden, model.hidden(q0))
    np.testing.assert_array_equal(last.logits, model.logits(q0))
    # Padded input slots are cleared even after a full nonzero batch.
    padding = dev.read(gpu.px + 9 * gpu.k1, (16 - 9) * gpu.k1)
    assert padding == bytes(len(padding))


@pytest.mark.parametrize("batch,reserved", [(9, 0), (None, 6000)])
def test_unfittable_tiled_batch_fails_before_upload(batch, reserved, monkeypatch):
    dev = Device(EmuBackend(Config(gmem_bytes=64 * 1024)))
    dev.alloc(reserved)
    def unexpected_write(*args):
        pytest.fail("an unfittable model must not send any data to the device")
    monkeypatch.setattr(dev.backend, "write_bytes", unexpected_write)
    with pytest.raises(MemoryError, match="only .* fit"):
        BatchMLP(dev, QuantMLP.load(), batch=batch)
    assert dev.memory_used == reserved


@pytest.mark.parametrize("batch", [0, -1, 1.5, True])
def test_tiled_batch_must_be_a_positive_integer(batch):
    with pytest.raises(ValueError, match="batch"):
        BatchMLP(Device(EmuBackend()), rounding_model(), batch=batch)


def test_stream_validates_later_inputs_before_starting(monkeypatch):
    gpu = BatchMLP(Device(EmuBackend()), rounding_model(), batch=8)
    q0 = np.zeros((9, 3), dtype=np.int32)
    q0[-1, -1] = -1
    def unexpected_write(*args):
        pytest.fail("invalid later inputs must be rejected before the first device write")
    monkeypatch.setattr(gpu.device, "write", unexpected_write)
    with pytest.raises(ValueError, match="input"):
        list(gpu.run_batches(q0))


@pytest.mark.parametrize("bad", ["shift", "shape", "weights", "overflow", "imem", "groups"])
def test_tiled_model_validation_precedes_device_writes(bad, monkeypatch):
    model = rounding_model()
    cfg = Config()
    if bad == "shift":
        model.sh1 = 0
    elif bad == "shape":
        model.b2 = np.zeros(2, dtype=np.int32)
    elif bad == "weights":
        model.w1 = model.w1.astype(float)
    elif bad == "overflow":
        model.b1[0] = (1 << 31) - 1
    elif bad == "imem":
        cfg.imem_words = 1
    else:
        model.w2 = np.zeros((65, 13), dtype=np.int8)
        model.b2 = np.zeros(65, dtype=np.int32)
    dev = Device(EmuBackend(cfg))
    def unexpected_write(*args):
        pytest.fail("invalid models must not reach the device")
    monkeypatch.setattr(dev.backend, "write_bytes", unexpected_write)
    with pytest.raises(ValueError):
        BatchMLP(dev, model, batch=8)
    assert dev.memory_used == 0


def test_broadcast_kernel_matches_and_saves_instructions():
    model = QuantMLP.load()
    q0 = np.random.default_rng(4).integers(0, 128, (8, model.w1.shape[1]), dtype=np.int32)
    plain, bcast = (BatchMLP(Device(EmuBackend()), model, batch=8, bcast=b).run(q0) for b in (False, True))
    for result in (plain, bcast):
        np.testing.assert_array_equal(result.hidden, model.hidden(q0))
        np.testing.assert_array_equal(result.logits, model.logits(q0))
    count = lambda r, k: sum(run.stats[k] for run in r.runs)  # noqa: E731
    assert count(bcast, "instructions") < 0.7 * count(plain, "instructions")
    assert count(bcast, "smem_wavefronts") < 0.25 * count(plain, "smem_wavefronts")


@pytest.mark.parametrize("cfg", [Config(dp4a=False), Config(num_lanes=4, num_warps=8)])
def test_broadcast_kernel_needs_dp4a_and_eight_lanes(cfg):
    with pytest.raises(ValueError, match="bcast"):
        BatchMLP(Device(EmuBackend(cfg)), rounding_model(), batch=8, bcast=True)


def test_double_buffered_tiles_match_and_need_bcast():
    model = QuantMLP.load()
    q0 = np.random.default_rng(6).integers(0, 128, (13, model.w1.shape[1]), dtype=np.int32)
    result = BatchMLP(Device(EmuBackend()), model, batch=13, bcast=True, dbuf=True).run(q0)
    np.testing.assert_array_equal(result.hidden, model.hidden(q0))
    np.testing.assert_array_equal(result.logits, model.logits(q0))
    with pytest.raises(ValueError, match="dbuf needs bcast"):
        BatchMLP(Device(EmuBackend()), model, batch=8, dbuf=True)
