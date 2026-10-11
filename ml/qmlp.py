"""The quantized MLP that runs on the GPU, and its bit-exact NumPy reference.

Network: 784 -> 64 -> 10, ReLU. Everything the GPU computes is integer:

    q0     = pixel >> 1                                  uint8 0..255 -> int 0..127
    acc1   = b1 + W1 . q0                                int8 x int8 -> int32
    q1     = clamp((acc1 + 2^(sh1-1)) >> sh1, 0, 127)    ReLU + requantize to int8
    logits = b2 + W2 . q1                                int32
    digit  = argmax(logits)                              ties -> lowest index

Requantization is a rounding shift because the activation scale is chosen as a
power-of-two multiple of (input scale x weight scale). That costs at most one
bit of activation range and saves a multiplier per output, a common trade in
small accelerators.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
N_IN, N_HID, N_OUT = 784, 64, 10


def preprocess(pixels: np.ndarray) -> np.ndarray:
    """uint8 images (..., 28, 28) -> int32 (..., 784) in 0..127."""
    return (pixels.reshape(*pixels.shape[:-2], N_IN) >> 1).astype(np.int32)


@dataclass
class QuantMLP:
    w1: np.ndarray   # int8  (64, 784)
    b1: np.ndarray   # int32 (64,)
    sh1: int         # requantization shift
    w2: np.ndarray   # int8  (10, 64)
    b2: np.ndarray   # int32 (10,)

    def hidden(self, q0: np.ndarray) -> np.ndarray:
        acc1 = q0.astype(np.int64) @ self.w1.T.astype(np.int64) + self.b1
        assert np.abs(acc1).max() < 2**31
        return np.clip((acc1 + (1 << (self.sh1 - 1))) >> self.sh1, 0, 127).astype(np.int32)

    def logits(self, q0: np.ndarray) -> np.ndarray:
        q1 = self.hidden(q0)
        return (q1.astype(np.int64) @ self.w2.T.astype(np.int64) + self.b2).astype(np.int32)

    def predict(self, pixels: np.ndarray) -> np.ndarray:
        return np.argmax(self.logits(preprocess(pixels)), axis=-1)

    def save(self, path: Path = ARTIFACTS / "mlp_int8.npz") -> None:
        path.parent.mkdir(exist_ok=True)
        np.savez(path, w1=self.w1, b1=self.b1, sh1=self.sh1, w2=self.w2, b2=self.b2)

    @classmethod
    def load(cls, path: Path = ARTIFACTS / "mlp_int8.npz") -> "QuantMLP":
        d = np.load(path)
        return cls(d["w1"].astype(np.int8), d["b1"].astype(np.int32), int(d["sh1"]),
                   d["w2"].astype(np.int8), d["b2"].astype(np.int32))


def quantize(w1, b1, w2, b2, calib_q0: np.ndarray, percentile: float = 99.99) -> QuantMLP:
    """Post-training quantization of a float model trained on inputs q0 / 127."""
    s0 = 1.0 / 127                                  # input scale
    sw1 = np.abs(w1).max() / 127
    w1q = np.clip(np.round(w1 / sw1), -127, 127).astype(np.int8)
    b1q = np.round(b1 / (s0 * sw1)).astype(np.int32)
    # activation range from calibration data -> power-of-two requant shift
    h1 = np.maximum(calib_q0 * s0 @ w1.T + b1, 0)
    s1_ideal = np.percentile(h1, percentile) / 127
    sh1 = int(np.round(np.log2(s1_ideal / (s0 * sw1))))
    s1 = s0 * sw1 * 2.0 ** sh1
    sw2 = np.abs(w2).max() / 127
    w2q = np.clip(np.round(w2 / sw2), -127, 127).astype(np.int8)
    b2q = np.round(b2 / (s1 * sw2)).astype(np.int32)
    return QuantMLP(w1q, b1q, sh1, w2q, b2q)
