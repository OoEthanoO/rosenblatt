"""Train the 784-64-10 digit classifier in PyTorch, quantize it to int8, and
check the integer model that the GPU will run.

    python ml/train.py            # ~1 minute on a laptop CPU

Writes ml/artifacts/mlp_fp32.pt and ml/artifacts/mlp_int8.npz.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mnist  # noqa: E402
from qmlp import ARTIFACTS, N_HID, N_IN, N_OUT, QuantMLP, preprocess, quantize  # noqa: E402


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_IN, N_HID)
        self.fc2 = nn.Linear(N_HID, N_OUT)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class FakeQuant(torch.autograd.Function):
    """Round in the forward pass, pass gradients straight through."""

    @staticmethod
    def forward(ctx, x, scale, lo, hi):
        return torch.clamp(torch.round(x / scale), lo, hi) * scale

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None


def qat_forward(model: MLP, x: torch.Tensor, sh1: int) -> torch.Tensor:
    """Forward pass that mimics the integer pipeline (weights int8, hidden int8 with a
    power-of-two scale) while keeping float gradients."""
    s0 = 1.0 / 127
    sw1 = model.fc1.weight.abs().max().detach() / 127
    w1 = FakeQuant.apply(model.fc1.weight, sw1, -127, 127)
    h = torch.relu(x @ w1.T + model.fc1.bias)
    s1 = s0 * sw1 * 2.0 ** sh1
    h = FakeQuant.apply(h, s1, 0, 127)
    sw2 = model.fc2.weight.abs().max().detach() / 127
    w2 = FakeQuant.apply(model.fc2.weight, sw2, -127, 127)
    return h @ w2.T + model.fc2.bias


def evaluate(model, x, y, fn=None) -> float:
    model.eval()
    with torch.no_grad():
        out = fn(x) if fn else model(x)
        return (out.argmax(1) == y).float().mean().item()


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    d = mnist.load()
    q_train, q_test = preprocess(d["train_images"]), preprocess(d["test_images"])
    x_train = torch.tensor(q_train / 127.0, dtype=torch.float32)
    x_test = torch.tensor(q_test / 127.0, dtype=torch.float32)
    y_train = torch.tensor(d["train_labels"], dtype=torch.long)
    y_test = torch.tensor(d["test_labels"], dtype=torch.long)

    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    epochs, batch = 15, 128
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * (60000 // batch + 1))
    loss_fn = nn.CrossEntropyLoss()
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(60000)
        for i in range(0, 60000, batch):
            idx = perm[i:i + batch]
            loss = loss_fn(model(x_train[idx]), y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        print(f"epoch {ep + 1:2d}  loss {loss.item():.4f}  test acc {evaluate(model, x_test, y_test):.4f}")
    acc_fp32 = evaluate(model, x_test, y_test)
    print(f"float32 model: {acc_fp32:.4f} test accuracy ({time.time() - t0:.0f}s)")

    def to_np(m):
        return (m.fc1.weight.detach().numpy().astype(np.float64), m.fc1.bias.detach().numpy().astype(np.float64),
                m.fc2.weight.detach().numpy().astype(np.float64), m.fc2.bias.detach().numpy().astype(np.float64))

    calib = q_train[:10000]
    qm = quantize(*to_np(model), calib)
    acc_ptq = (qm.predict(d["test_images"]) == d["test_labels"]).mean()
    print(f"int8 post-training quantization: {acc_ptq:.4f} (sh1 = {qm.sh1})")

    # short quantization-aware fine-tuning with the shift fixed
    sh1 = qm.sh1
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    for ep in range(3):
        model.train()
        perm = torch.randperm(60000)
        for i in range(0, 60000, batch):
            idx = perm[i:i + batch]
            loss = loss_fn(qat_forward(model, x_train[idx], sh1), y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        qa = evaluate(model, x_test, y_test, lambda x: qat_forward(model, x, sh1))
        print(f"QAT epoch {ep + 1}: fake-quant test acc {qa:.4f}")
    qm_qat = quantize(*to_np(model), calib)
    acc_qat = (qm_qat.predict(d["test_images"]) == d["test_labels"]).mean()
    print(f"int8 after QAT: {acc_qat:.4f} (sh1 = {qm_qat.sh1})")

    best = qm_qat if acc_qat >= acc_ptq else qm
    ARTIFACTS.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ARTIFACTS / "mlp_fp32.pt")
    best.save()
    acc = (best.predict(d["test_images"]) == d["test_labels"]).mean()
    (ARTIFACTS / "accuracy.txt").write_text(
        f"float32 {acc_fp32:.4f}\nint8_ptq {acc_ptq:.4f}\nint8_qat {acc_qat:.4f}\nint8_deployed {acc:.4f}\n")
    print(f"saved {ARTIFACTS / 'mlp_int8.npz'}: deployed int8 accuracy {acc:.4f}")


if __name__ == "__main__":
    main()
