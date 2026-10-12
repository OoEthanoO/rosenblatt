"""Board benchmark: real kernels, bounded memory, counter accounting and failed reports."""

import json
from contextlib import contextmanager

import numpy as np
import pytest

from bench import board
from ml.gpu import BatchMLP, GpuMLP
from ml.qmlp import QuantMLP
from rosenblatt.device import Device, EmuBackend
from rosenblatt.emu import Config


def workload():
    rng = np.random.default_rng(22)
    model = QuantMLP(rng.integers(-127, 128, (5, 3), dtype=np.int16).astype(np.int8),
                     np.zeros(5, dtype=np.int32), 8,
                     rng.integers(-127, 128, (2, 5), dtype=np.int16).astype(np.int8),
                     np.zeros(2, dtype=np.int32))
    inputs = rng.integers(0, 128, (19, 3), dtype=np.int32)
    labels = np.argmax(model.logits(inputs), axis=1)
    return model, inputs, labels


@pytest.mark.parametrize("timed", [False, True])
def test_benchmark_counts_real_images_and_separates_timing(timed):
    class Backend(EmuBackend):
        def launch(self, *args):
            result = super().launch(*args)
            if timed:
                result.cycles = 123  # synthetic counter for testing aggregation only
            return result

    dev = Device(Backend(Config(gmem_bytes=64 * 1024)))
    result = board.run_benchmark(dev, *workload(), batch=8, clock_hz=25_000_000, start_index=7)
    single, tiled = result["strategies"]
    for strategy in result["strategies"]:
        assert strategy["status"] == "passed"
        assert strategy["verified_images"] == strategy["correct"] == 19
        assert strategy["allocated_bytes"] < 64 * 1024
        assert strategy["setup_wall_seconds"] >= 0
        assert strategy["inference_wall_seconds"] > 0
        assert strategy["verification_wall_seconds"] > 0
        if not timed:
            assert strategy["kernel_cycles"] is None
            assert strategy["kernel_seconds"] is None
    assert [chunk["start_index"] for chunk in tiled["chunks"]] == [7, 15, 23]
    assert [chunk["count"] for chunk in tiled["chunks"]] == [8, 8, 3]
    assert tiled["padded_images"] == 24 and single["padded_images"] == 19
    if timed:
        assert single["kernel_cycles"] == 123 * 2 * 19
        assert tiled["kernel_cycles"] == 123 * 2 * 3
        assert tiled["cycles_per_image"] == 738 / 19
        assert tiled["kernel_seconds"] == 738 / 25_000_000
        assert result["kernel_speedup"] == 19 / 3
    else:
        assert result["kernel_speedup"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("layer", ["hidden", "logits"])
def test_benchmark_stops_on_hidden_or_nonwinning_logit_mismatch(layer, monkeypatch):
    real_run = BatchMLP.run
    calls = 0
    def corrupt(self, inputs):
        nonlocal calls
        calls += 1
        result = real_run(self, inputs)
        if calls == 2:
            winner = result.predictions[0]
            index = 0 if layer == "hidden" else (winner + 1) % 2
            getattr(result, layer)[0, index] -= 1
            assert result.predictions[0] == winner
        return result
    monkeypatch.setattr(BatchMLP, "run", corrupt)
    report = {}
    with pytest.raises(RuntimeError, match=f"image 15: {layer}"):
        board.run_benchmark(Device(EmuBackend()), *workload(), batch=8, start_index=7, report=report)
    assert report["strategies"][0]["status"] == "passed"
    failed = report["strategies"][1]
    assert failed["status"] == "failed" and failed["verified_images"] == 8
    assert calls == 2  # no retries and no third chunk


@pytest.mark.parametrize("failure", ["connect", "verify", "interrupt"])
def test_cli_overwrites_stale_success_and_closes_connection(failure, tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    output.write_text('{"status": "passed", "old": true}')
    monkeypatch.setattr(board.mnist, "load", lambda _: {
        "test_images": np.zeros((1, 28, 28), dtype=np.uint8),
        "test_labels": np.zeros(1, dtype=np.uint8),
    })
    closed = []
    @contextmanager
    def connection(*args, **kwargs):
        assert json.loads(output.read_text())["status"] == "running"
        if failure == "connect":
            raise TimeoutError("serial response timed out")
        try:
            yield EmuBackend(Config(gmem_bytes=64 * 1024))
        finally:
            closed.append(True)
    def broken(*args, **kwargs):
        if failure == "interrupt":
            raise KeyboardInterrupt
        raise RuntimeError("image 0: hidden[0] differs")
    monkeypatch.setattr(board, "UartBackend", connection)
    monkeypatch.setattr(board, "run_benchmark", broken)
    code = board.main(["--port", "TEST", "--count", "1", "--output", str(output)])
    report = json.loads(output.read_text())
    assert code == (130 if failure == "interrupt" else 1)
    assert report["status"] == ("interrupted" if failure == "interrupt" else "failed")
    assert "old" not in report and "error" in report and "finished_utc" in report
    assert closed == ([] if failure == "connect" else [True])


def test_cli_emulator_report_is_explicit_and_has_no_gpu_timing(tmp_path, monkeypatch):
    output = tmp_path / "nested" / "report.json"
    monkeypatch.setattr(board.mnist, "load", lambda _: {
        "test_images": np.zeros((9, 28, 28), dtype=np.uint8),
        "test_labels": np.zeros(9, dtype=np.uint8),
    })
    assert board.main(["--count", "9", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["status"] == "passed" and report["backend"] == "emulator"
    assert report["clock_hz"] is None and report["kernel_speedup"] is None
    assert report["strategies"][1]["batch_capacity"] == 8
    assert report["strategies"][1]["allocated_bytes"] == 60_416
    assert len(report["input_sha256"]) == len(report["model_file_sha256"]) == 64
    assert not list(output.parent.glob("*.tmp"))


def test_report_identity_changes_with_workload():
    model, inputs, labels = workload()
    dev = Device(EmuBackend())
    first = board.run_benchmark(dev, model, inputs[:1], labels[:1], batch=8)
    model.b1[0] += 1
    inputs[0, 0] += 1
    second = board.run_benchmark(dev, model, inputs[:1], labels[:1], batch=8)
    assert first["model"] != second["model"]
    assert first["input_sha256"] != second["input_sha256"]
