"""Draw a digit with the mouse; the GPU recognizes it.

    python -m ml.draw                 # on the emulator
    python -m ml.draw --port COM5     # on the ULX3S (the LEDs show the answer too)

Left mouse button draws, the right button (or Clear) wipes the canvas. When the
mouse is released, the drawing is turned into a 28x28 image the way MNIST's
digits were made, the GPU runs the network, and the window shows the digit,
all ten output scores, and the image the network actually saw. Every result is
checked against the NumPy reference before it is shown.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np

CANVAS = 280      # drawing area in pixels (10 per MNIST pixel)
BRUSH = 11        # brush radius in canvas pixels


# ------------------------------------------------------------- preprocessing


def rasterize(strokes: list[list[tuple[int, int]]], size: int = CANVAS, radius: int = BRUSH) -> np.ndarray:
    """Render polyline strokes as white-on-black uint8 with round brush tips."""
    img = np.zeros((size, size), dtype=np.uint8)
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    disc = (xx * xx + yy * yy) <= radius * radius

    def stamp(x: float, y: float) -> None:
        x0, y0 = int(round(x)) - radius, int(round(y)) - radius
        xs, ys = slice(max(x0, 0), min(x0 + 2 * radius + 1, size)), slice(max(y0, 0), min(y0 + 2 * radius + 1, size))
        if xs.start >= xs.stop or ys.start >= ys.stop:
            return
        d = disc[ys.start - y0: ys.stop - y0, xs.start - x0: xs.stop - x0]
        img[ys, xs][d] = 255

    for stroke in strokes:
        if len(stroke) == 1:
            stamp(*stroke[0])
        for (xa, ya), (xb, yb) in zip(stroke, stroke[1:]):
            steps = max(1, int(np.hypot(xb - xa, yb - ya)))
            for t in np.linspace(0.0, 1.0, steps + 1):
                stamp(xa + (xb - xa) * t, ya + (yb - ya) * t)
    return img


def _area_resize(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Shrink by averaging the source pixels that fall into each target pixel."""
    ys = np.linspace(0, img.shape[0], h + 1).round().astype(int)
    xs = np.linspace(0, img.shape[1], w + 1).round().astype(int)
    out = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            block = img[ys[i]:max(ys[i + 1], ys[i] + 1), xs[j]:max(xs[j + 1], xs[j] + 1)]
            out[i, j] = block.mean()
    return out


def to_mnist(img: np.ndarray) -> np.ndarray:
    """A white-on-black drawing of any size -> 28x28 uint8, normalized like MNIST:
    the digit's bounding box is scaled to fit 20x20 (keeping its aspect ratio)
    and its center of mass is placed at the center of the 28x28 frame."""
    ys, xs = np.nonzero(img > 0)
    out = np.zeros((28, 28), dtype=np.uint8)
    if ys.size == 0:
        return out
    crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float64)
    scale = 20.0 / max(crop.shape)
    h, w = max(1, round(crop.shape[0] * scale)), max(1, round(crop.shape[1] * scale))
    small = _area_resize(crop, h, w)
    total = small.sum()
    cy = (small.sum(axis=1) * np.arange(h)).sum() / total
    cx = (small.sum(axis=0) * np.arange(w)).sum() / total
    oy = int(np.clip(round(13.5 - cy), 0, 28 - h))
    ox = int(np.clip(round(13.5 - cx), 0, 28 - w))
    out[oy:oy + h, ox:ox + w] = np.clip(small, 0, 255).round().astype(np.uint8)
    return out


# --------------------------------------------------------------------- GUI


def checked_inference(gpu, model, image: np.ndarray):
    """Reject a corrupted intermediate layer even when the final digit agrees."""
    from .qmlp import preprocess

    q0 = preprocess(image)
    result = gpu.run(q0)
    if not np.array_equal(result.hidden, model.hidden(q0)):
        raise RuntimeError("GPU hidden activations differ from the reference")
    if not np.array_equal(result.logits, model.logits(q0)):
        raise RuntimeError("GPU output scores differ from the reference")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port of the ULX3S (default: the emulator)")
    ap.add_argument("--baudrate", type=int, default=115200)
    ap.add_argument("--selftest", action="store_true", help=argparse.SUPPRESS)  # hidden window, for tests
    a = ap.parse_args(argv)

    import tkinter as tk

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sw"))
    from rosenblatt import isa
    from rosenblatt.device import Device, EmuBackend

    from .gpu import GpuMLP
    from .qmlp import QuantMLP

    model = QuantMLP.load()
    if a.port:
        from rosenblatt.uart import UartBackend
        connection = UartBackend(a.port, baudrate=a.baudrate)
    else:
        connection = nullcontext(EmuBackend())

    with connection as backend:
        dev = Device(backend)
        gpu = GpuMLP(dev, model)
        where = f"ULX3S on {a.port}" if a.port else "emulator"

        root = tk.Tk()
        root.title("Rosenblatt: draw a digit")
        root.configure(bg="#111827")
        fg, dim, accent = "#f9fafb", "#9ca3af", "#60a5fa"

        board = tk.Canvas(root, width=CANVAS, height=CANVAS, bg="black", highlightthickness=0,
                          cursor="crosshair")
        board.grid(row=0, column=0, rowspan=4, padx=16, pady=16)
        side = tk.Frame(root, bg="#111827")
        side.grid(row=0, column=1, sticky="n", padx=(0, 16), pady=16)
        digit = tk.Label(side, text="?", font=("Segoe UI", 64, "bold"), fg=fg, bg="#111827", width=2)
        digit.pack()
        status = tk.Label(side, text=f"GPU: {where}\ndraw a digit", font=("Segoe UI", 10),
                          fg=dim, bg="#111827", justify="left", wraplength=220)
        status.pack(anchor="w")
        bars = tk.Canvas(side, width=220, height=150, bg="#111827", highlightthickness=0)
        bars.pack(pady=(8, 4))
        seen = tk.Canvas(side, width=112, height=112, bg="black", highlightthickness=0)
        seen.pack(pady=4)
        tk.Label(side, text="what the network sees (28x28)", font=("Segoe UI", 8), fg=dim,
                 bg="#111827").pack()

        strokes: list[list[tuple[int, int]]] = []
        failed = False

        def press(e):
            strokes.append([(e.x, e.y)])
            board.create_oval(e.x - BRUSH, e.y - BRUSH, e.x + BRUSH, e.y + BRUSH, fill="white", outline="")

        def drag(e):
            if not strokes:
                return press(e)
            x0, y0 = strokes[-1][-1]
            strokes[-1].append((e.x, e.y))
            board.create_line(x0, y0, e.x, e.y, fill="white", width=2 * BRUSH, capstyle="round")

        def clear(_=None):
            strokes.clear()
            board.delete("all")
            seen.delete("all")
            bars.delete("all")
            if not failed:
                digit.config(text="?")
                status.config(text=f"GPU: {where}\ndraw a digit")

        def recognize(_=None):
            nonlocal failed
            if not strokes or failed:
                return
            try:
                img28 = to_mnist(rasterize(strokes))
                res = checked_inference(gpu, model, img28)
                if a.port:
                    backend.bus_write(isa.HOST_BOARD_BASE, [res.prediction])
            except Exception as exc:
                # A failed UART transaction must never be retried: the FPGA
                # may already have performed it. Require a fresh connection.
                failed = True
                digit.config(text="!", fg="#ef4444")
                bars.delete("all")
                seen.delete("all")
                recovery = "Press FIRE1 and restart this demo." if a.port else "Restart this demo."
                status.config(text=f"GPU: {where}\n{exc}\n{recovery}", fg="#ef4444")
                return
            pred = res.prediction
            cycles = sum(r.cycles or 0 for r in res.runs)
            instr = sum(r.stats.get("instructions", 0) for r in res.runs)
            digit.config(text=str(pred), fg=fg)
            work = f"{cycles:,} GPU cycles" if cycles else f"{instr:,} warp instructions"
            status.config(text=f"GPU: {where}\n{work}\nmatches the NumPy reference")
            # the ten output scores, scaled to the strongest
            bars.delete("all")
            s = res.logits.astype(float)
            span = max(1.0, s.max() - min(0.0, s.min()))
            for k in range(10):
                x = 6 + k * 21
                hgt = max(1, int(110 * (s[k] - min(0.0, s.min())) / span))
                bars.create_rectangle(x, 120 - hgt, x + 15, 120, fill=accent if k == pred else "#374151",
                                      outline="")
                bars.create_text(x + 7, 135, text=str(k), fill=fg if k == pred else dim,
                                 font=("Segoe UI", 9))
            seen.delete("all")
            for yy in range(28):
                for xx in range(28):
                    v = int(img28[yy, xx])
                    if v:
                        c = f"#{v:02x}{v:02x}{v:02x}"
                        seen.create_rectangle(xx * 4, yy * 4, xx * 4 + 4, yy * 4 + 4, fill=c, outline="")

        board.bind("<Button-1>", press)
        board.bind("<B1-Motion>", drag)
        board.bind("<ButtonRelease-1>", recognize)
        board.bind("<Button-3>", clear)
        tk.Button(side, text="Clear", command=clear, bg="#374151", fg=fg, relief="flat",
                  activebackground="#4b5563", activeforeground=fg).pack(fill="x", pady=(8, 0))
        if a.selftest:   # drive the real handlers without showing a window
            root.withdraw()
            strokes.append([(140, 40), (140, 240)])
            recognize()
            print(f"selftest: drew a vertical stroke, GPU says {digit.cget('text')}; "
                  f"{status.cget('text').splitlines()[-1]}")
            root.destroy()
            return int(failed)
        root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
