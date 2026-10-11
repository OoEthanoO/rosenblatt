"""Fast inference checks that do not require an RTL simulator or downloads."""

import numpy as np
import pytest

from ml.gpu import BatchMLP, GpuMLP
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
