"""Run both layers of a QuantMLP through the Rosenblatt Device runtime.

Weights and scratch buffers stay on the device across calls. Only the input
is uploaded per inference; the second layer reads the first layer's output
directly. The host reads activations/logits for verification and picks argmax.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rosenblatt.asm import assemble_file, link
from rosenblatt.device import Device, LaunchResult

from .qmlp import QuantMLP

KERNEL = Path(__file__).resolve().parents[1] / "kernels" / "dense_int8.s"


@dataclass
class InferenceResult:
    hidden: np.ndarray
    logits: np.ndarray
    runs: list[LaunchResult]

    @property
    def prediction(self) -> int:
        """Equal logits choose the lowest digit index, as in the reference."""
        return int(np.argmax(self.logits))


def _integer_array(value, name: str, lo: int, hi: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iu" or np.any(array < lo) or np.any(array > hi):
        raise ValueError(f"{name} must contain integers in {lo}..{hi}")
    return array


def _check_accumulator(weights: np.ndarray, bias: np.ndarray, rounding: int) -> None:
    """Bound every partial sum for inputs in 0..127, including rounding."""
    weights, bias = weights.astype(np.int64), bias.astype(np.int64)
    low = bias + 127 * np.minimum(weights, 0).sum(axis=1)
    high = bias + 127 * np.maximum(weights, 0).sum(axis=1) + rounding
    if np.any(low < -(1 << 31)) or np.any(high >= 1 << 31):
        raise ValueError("model may overflow a signed int32 accumulator")


class GpuMLP:
    """A two-layer int8 MLP with the same arithmetic as QuantMLP.

    Supports arbitrary positive layer widths that fit device memory. Input
    and hidden widths are padded to four bytes. A device with a 16-bit
    multiplier requires padded widths and output indices below 32768.
    Do not reset the device heap while this object is in use.
    """

    def __init__(self, device: Device, model: QuantMLP, *, dp4a: bool | None = None):
        self.device = device
        self.dp4a = device.cfg.dp4a if dp4a is None else dp4a
        if self.dp4a and not device.cfg.dp4a:
            raise ValueError("DP4A is not supported by this device")
        w1 = _integer_array(model.w1, "w1", -128, 127)
        w2 = _integer_array(model.w2, "w2", -128, 127)
        b1 = _integer_array(model.b1, "b1", -(1 << 31), (1 << 31) - 1)
        b2 = _integer_array(model.b2, "b2", -(1 << 31), (1 << 31) - 1)
        if w1.ndim != 2 or w2.ndim != 2 or min(*w1.shape, *w2.shape) <= 0:
            raise ValueError("weights must be nonempty matrices")
        self.n_hidden, self.n_input = w1.shape
        self.n_output, hidden = w2.shape
        if hidden != self.n_hidden or b1.shape != (hidden,) or b2.shape != (self.n_output,):
            raise ValueError("weight and bias shapes do not form a two-layer MLP")
        if not isinstance(model.sh1, (int, np.integer)) or not 1 <= model.sh1 <= 31:
            raise ValueError("hidden requantization shift must be an integer in 1..31")
        self.shift = int(model.sh1)
        _check_accumulator(w1, b1, 1 << (self.shift - 1))
        _check_accumulator(w2, b2, 0)

        self.input_width = (self.n_input + 3) // 4 * 4
        self.hidden_width = (self.n_hidden + 3) // 4 * 4
        if device.cfg.mul_width == 16 and max(
            self.input_width, self.hidden_width, self.n_hidden - 1, self.n_output - 1
        ) >= 32768:
            raise ValueError("layer dimensions exceed the 16-bit multiplier's address arithmetic")

        hidden_program = assemble_file(str(KERNEL), defines={"DP4A": int(self.dp4a), "REQUANTIZE": 1})
        output_program = assemble_file(str(KERNEL), base=hidden_program.end,
                                       defines={"DP4A": int(self.dp4a), "REQUANTIZE": 0})
        self.program = link({"hidden": hidden_program, "output": output_program})
        if len(self.program) > device.cfg.imem_words:
            raise ValueError("MLP kernels do not fit device instruction memory")

        def upload_weights(weights, width):
            padded = np.zeros((weights.shape[0], width), dtype=np.int8)
            padded[:, :weights.shape[1]] = weights
            return device.upload(padded)

        self.pw1 = upload_weights(w1, self.input_width)
        self.pb1 = device.upload(b1.astype(np.int32))
        self.pw2 = upload_weights(w2, self.hidden_width)
        self.pb2 = device.upload(b2.astype(np.int32))
        self.pinput = device.upload(np.zeros(self.input_width, dtype=np.int8))
        self.phidden = device.upload(np.zeros(self.hidden_width, dtype=np.int8))
        self.poutput = device.alloc(self.n_output * 4)

    def run(self, q0: np.ndarray) -> InferenceResult:
        """Infer one preprocessed input vector, with integer values in 0..127."""
        q0 = _integer_array(q0, "input", 0, 127)
        if q0.shape != (self.n_input,):
            raise ValueError(f"input must have shape ({self.n_input},), got {q0.shape}")
        padded = np.zeros(self.input_width, dtype=np.int8)
        padded[:self.n_input] = q0
        dev = self.device
        dev.write(self.pinput, padded)
        dev.load(self.program)
        block = dev.cfg.max_threads
        runs = [dev.launch(
            grid=(self.n_hidden + block - 1) // block, block=block, entry="hidden",
            args=[self.pinput, self.pw1, self.pb1, self.phidden, self.input_width,
                  self.n_hidden, self.shift],
        ), dev.launch(
            grid=(self.n_output + block - 1) // block, block=block, entry="output",
            args=[self.phidden, self.pw2, self.pb2, self.poutput, self.hidden_width,
                  self.n_output],
        )]
        return InferenceResult(dev.download(self.phidden, self.n_hidden, np.int8).astype(np.int32),
                               dev.download(self.poutput, self.n_output), runs)


TILED_KERNEL = Path(__file__).resolve().parents[1] / "kernels" / "dense_tiled.s"


@dataclass
class BatchResult:
    hidden: np.ndarray   # (batch, n_hidden)
    logits: np.ndarray   # (batch, n_output)
    runs: list[LaunchResult]

    @property
    def predictions(self) -> np.ndarray:
        return np.argmax(self.logits, axis=1)


class BatchMLP:
    """Batched inference through the shared-memory tiled kernel (kernels/dense_tiled.s).

    Each block computes THREADS neurons for BT = 8 images and stages the
    images' inputs in shared memory, so every weight fetched from global
    memory is reused for 8 images instead of 1. This is the kernel the
    profiling study compares against GpuMLP's one-image-at-a-time approach.

    Needs blocks of exactly 32 threads (the default 8 lanes x 4 warps).
    """

    BT = 8    # images per block
    KC = 16   # input bytes staged per step (BT * KC / 4 == 32 threads)

    def __init__(self, device: Device, model: QuantMLP, batch: int, *, dp4a: bool | None = None):
        cfg = device.cfg
        self.threads = cfg.max_threads
        if self.threads != self.BT * self.KC // 4:
            raise ValueError(f"BatchMLP needs {self.BT * self.KC // 4}-thread blocks, device has {self.threads}")
        self.device = device
        self.dp4a = cfg.dp4a if dp4a is None else dp4a
        if self.dp4a and not cfg.dp4a:
            raise ValueError("DP4A is not supported by this device")
        self.model = model
        n_hidden, n_input = model.w1.shape
        n_output = model.w2.shape[0]
        self.n_input, self.n_hidden, self.n_output = n_input, n_hidden, n_output
        _check_accumulator(np.asarray(model.w1), np.asarray(model.b1), 1 << (int(model.sh1) - 1))
        _check_accumulator(np.asarray(model.w2), np.asarray(model.b2), 0)
        up = lambda n, m: (n + m - 1) // m * m  # noqa: E731
        self.k1 = up(n_input, self.KC)                 # padded input width
        self.n1 = up(n_hidden, self.threads)           # padded hidden width = layer-2 input width
        self.n2 = up(n_output, self.threads)
        if self.n1 % self.KC:
            raise ValueError("padded hidden width must be a multiple of KC")
        self.batch = up(batch, self.BT)

        def pad2(a, rows, cols):
            out = np.zeros((rows, cols), dtype=np.int8)
            out[:a.shape[0], :a.shape[1]] = a
            return out

        def pad1(a, n):
            out = np.zeros(n, dtype=np.int32)
            out[:a.shape[0]] = a
            return out

        self.pw1 = device.upload(pad2(model.w1, self.n1, self.k1))
        self.pb1 = device.upload(pad1(model.b1, self.n1))
        self.pw2 = device.upload(pad2(model.w2, self.n2, self.n1))
        self.pb2 = device.upload(pad1(model.b2, self.n2))
        self.px = device.alloc(self.batch * self.k1)
        self.ph = device.alloc(self.batch * self.n1)
        self.py = device.alloc(self.batch * self.n2 * 4)

        def kernel(n_pad: int, requantize: bool, base: int):
            groups = n_pad // self.threads
            return assemble_file(str(TILED_KERNEL), base=base, defines={
                "THREADS": self.threads, "BT": self.BT, "KC": self.KC,
                "LOG_KC": self.KC.bit_length() - 1, "LOG_KC_WORDS": (self.KC // 4).bit_length() - 1,
                "LOG_NGROUPS": groups.bit_length() - 1, "DP4A": int(self.dp4a),
                "REQUANTIZE": int(requantize)})

        for n in (self.n1, self.n2):
            groups = n // self.threads
            if groups & (groups - 1):
                raise ValueError("padded layer widths must be a power-of-two number of blocks")
        hidden = kernel(self.n1, True, 0)
        output = kernel(self.n2, False, hidden.end)
        self.program = link({"hidden": hidden, "output": output})
        if len(self.program) > cfg.imem_words:
            raise ValueError("tiled kernels do not fit device instruction memory")

    def run(self, q0: np.ndarray) -> BatchResult:
        """Infer a batch of preprocessed inputs, shape (n, n_input), values 0..127."""
        q0 = _integer_array(q0, "input", 0, 127)
        if q0.ndim != 2 or q0.shape[1] != self.n_input or not 0 < q0.shape[0] <= self.batch:
            raise ValueError(f"input must have shape (1..{self.batch}, {self.n_input})")
        n = q0.shape[0]
        x = np.zeros((self.batch, self.k1), dtype=np.int8)
        x[:n, :self.n_input] = q0
        dev = self.device
        dev.write(self.px, x)
        dev.load(self.program)
        blocks = self.batch // self.BT
        shift = int(self.model.sh1)
        runs = [
            dev.launch(grid=blocks * (self.n1 // self.threads), block=self.threads, entry="hidden",
                       args=[self.px, self.pw1, self.pb1, self.ph, self.k1, self.n1, shift]),
            dev.launch(grid=blocks * (self.n2 // self.threads), block=self.threads, entry="output",
                       args=[self.ph, self.pw2, self.pb2, self.py, self.n1, self.n2, 0]),
        ]
        hidden = dev.download(self.ph, self.batch * self.n1, np.int8).reshape(self.batch, self.n1)
        logits = dev.download(self.py, self.batch * self.n2, np.int32).reshape(self.batch, self.n2)
        return BatchResult(hidden[:n, :self.n_hidden].astype(np.int32), logits[:n, :self.n_output], runs)
