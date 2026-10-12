"""The board bring-up checks and the drawing demo's preprocessing (no hardware, no window)."""

import sys
from pathlib import Path

import numpy as np
import pytest

from rosenblatt.device import Device, EmuBackend
from rosenblatt.emu import Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fpga"))
import bringup  # noqa: E402
from ml.draw import checked_inference, rasterize, to_mnist  # noqa: E402
from ml.qmlp import QuantMLP  # noqa: E402


@pytest.mark.parametrize("cfg", [Config(), Config(num_cores=1, num_lanes=4, mul_width=16)])
def test_bringup_checks_pass_on_the_emulator(cfg, capsys):
    bringup.selftest(Device(EmuBackend(cfg)), images=0)
    out = capsys.readouterr().out
    assert out.count("PASS") == 4 and "FAIL" not in out


def test_bringup_reports_the_failing_layer(capsys, monkeypatch):
    dev = Device(EmuBackend())
    real_read = dev.read
    monkeypatch.setattr(dev, "read", lambda addr, n: bytes(n) if addr == 0 else real_read(addr, n))
    with pytest.raises(bringup.StepFailed):
        bringup.selftest(dev, images=0)
    assert "FAIL  global memory" in capsys.readouterr().out


def test_drawing_is_normalized_like_mnist():
    img = to_mnist(rasterize([[(140, 40), (140, 240)]]))
    assert img.shape == (28, 28) and img.dtype == np.uint8
    ys, xs = np.nonzero(img)
    assert ys.max() - ys.min() + 1 == 20            # the long side is scaled to 20 pixels
    cy = (img.sum(axis=1) * np.arange(28)).sum() / img.sum()
    cx = (img.sum(axis=0) * np.arange(28)).sum() / img.sum()
    assert abs(cy - 13.5) < 1 and abs(cx - 13.5) < 1   # centered by mass
    assert not to_mnist(np.zeros((280, 280), dtype=np.uint8)).any()


def test_drawn_strokes_are_recognized():
    model = QuantMLP.load()
    one = to_mnist(rasterize([[(140, 40), (140, 240)]]))
    t = np.linspace(0, 2 * np.pi, 60)
    zero = to_mnist(rasterize([[(140 + int(70 * np.cos(a)), 140 + int(100 * np.sin(a))) for a in t]]))
    assert model.predict(np.stack([one, zero])).tolist() == [1, 0]


@pytest.mark.parametrize("corrupt", [None, "hidden", "logits"])
def test_drawing_checks_both_layers_before_showing_a_prediction(corrupt, monkeypatch):
    from ml.gpu import GpuMLP

    model = QuantMLP.load()
    gpu = GpuMLP(Device(EmuBackend(Config(gmem_bytes=64 * 1024))), model)
    image = to_mnist(rasterize([[(140, 40), (140, 240)]]))
    real_run = gpu.run

    def run(q0):
        result = real_run(q0)
        if corrupt:
            # These faults preserve the winning digit. Checking only the
            # prediction (or only the logits) would silently accept them.
            original = result.prediction
            index = 0 if corrupt == "hidden" else (original + 1) % 10
            getattr(result, corrupt)[index] -= 1
            assert result.prediction == original
        return result

    monkeypatch.setattr(gpu, "run", run)
    if corrupt:
        with pytest.raises(RuntimeError, match="differ from the reference"):
            checked_inference(gpu, model, image)
    else:
        assert checked_inference(gpu, model, image).prediction == 1
