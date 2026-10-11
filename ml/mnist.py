"""MNIST loader: downloads the four IDX files once (torchvision's mirror) and
returns NumPy arrays. No torchvision dependency."""

from __future__ import annotations

import gzip
import hashlib
import urllib.request
from pathlib import Path

import numpy as np

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_images": ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    "train_labels": ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    "test_images": ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    "test_labels": ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
}
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "mnist"


def _fetch(name: str, md5: str) -> Path:
    path = DATA_DIR / name
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        print(f"downloading {MIRROR + name}")
        urllib.request.urlretrieve(MIRROR + name, tmp)
        tmp.replace(path)
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    if digest != md5:
        raise RuntimeError(f"{path} has md5 {digest}, expected {md5}; delete it and retry")
    return path


def _read_idx(path: Path) -> np.ndarray:
    raw = gzip.decompress(path.read_bytes())
    ndim = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load(split: str | None = None) -> dict[str, np.ndarray]:
    """Load both splits, or only 'train' / 'test', downloading missing files.

    Keys are train_images, train_labels, test_images, test_labels; images
    have shape (N, 28, 28) and all arrays have dtype uint8.
    """
    if split not in (None, "train", "test"):
        raise ValueError("split must be 'train', 'test', or None")
    return {k: _read_idx(_fetch(*v)) for k, v in FILES.items()
            if split is None or k.startswith(split + "_")}


if __name__ == "__main__":
    d = load()
    for k, v in d.items():
        print(f"{k:13s} {v.shape} {v.dtype}")
