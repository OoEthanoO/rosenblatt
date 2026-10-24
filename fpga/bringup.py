"""First contact with a real board: find it, load the GPU, test it layer by layer.

    python fpga/bringup.py                    # find the ULX3S, program it, run every check
    python fpga/bringup.py --board basys3     # the same for a Digilent Basys 3
    python fpga/bringup.py --port COM5 --no-program
    python fpga/bringup.py --list             # show the serial ports this PC sees
    python fpga/bringup.py --doctor           # check software before the board arrives

Every step prints PASS or FAIL with a hint and the run stops at the first
failure, so a problem is pinned to one layer: USB cable or driver, bitstream,
serial link, global memory, a simple kernel, barriers and shared memory, the
full digit network. The checks themselves (`selftest`) take any Device, so
they also run on the emulator: `python fpga/bringup.py --emulator`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "sw"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from rosenblatt import isa  # noqa: E402
from rosenblatt.asm import assemble_file  # noqa: E402
from rosenblatt.device import BusBackend, Device, EmuBackend  # noqa: E402
from rosenblatt.emu import Config  # noqa: E402
from fpga import build as fpga_build  # noqa: E402
from fpga import vivado_build  # noqa: E402

KERNELS = ROOT / "kernels"
FTDI_VID, FT231X_PID = 0x0403, 0x6015   # the ULX3S's USB-serial chip
# What differs between the supported boards; the checks themselves are shared.
BOARDS = {
    "ulx3s-85f": {"name": "ULX3S", "usb": (FTDI_VID, FT231X_PID), "chip": "FT231X",
                  "connector": "the port marked US1", "reset": "FIRE1"},
    "basys3": {"name": "Basys 3", "usb": (FTDI_VID, 0x6010), "chip": "FT2232H",
               "connector": "the port marked PROG/UART", "reset": "BTNC", "part": "xc7a35t"},
}


def build_command(board: str, revision: str = fpga_build.DEFAULT_REVISION) -> str:
    if board == "basys3":
        return "python fpga/vivado_build.py basys3"
    return "python fpga/build.py ulx3s-85f --revision " + revision


def default_bitstream(board: str, revision: str = fpga_build.DEFAULT_REVISION) -> Path:
    if board == "basys3":
        return vivado_build.output_dir("basys3") / "top.bit"
    return fpga_build.output_dir("ulx3s-85f", revision=revision) / "top.bit"


class StepFailed(Exception):
    pass


def step(name: str, hint: str):
    """step(name, hint)(fn): run fn, print PASS or FAIL, and stop the run on failure."""
    def run(fn):
        t = time.perf_counter()
        try:
            detail = fn()
        except Exception as e:  # noqa: BLE001 - report everything, stop the run
            print(f"FAIL  {name}: {e}\n      hint: {hint}")
            raise StepFailed(name) from e
        print(f"PASS  {name} ({time.perf_counter() - t:.1f}s){': ' + detail if detail else ''}")
    return run


# ------------------------------------------------------------------ the checks


def selftest(dev: Device, *, images: int = 20, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    cfg = dev.cfg

    def config():
        return (f"{cfg.num_cores} cores x {cfg.num_lanes} lanes x {cfg.num_warps} warps, "
                f"{cfg.gmem_bytes // 1024} KiB global memory, DP4A {'on' if cfg.dp4a else 'off'}")
    step("hardware configuration", "the bitstream and this code may be from different versions")(config)

    def memory():
        n = cfg.gmem_bytes
        data = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
        dev.write(0, data)
        if dev.read(0, n) != data:
            raise AssertionError("random data did not read back")
        # one word at every power-of-two address catches stuck or shorted address lines
        addrs = [0] + [1 << k for k in range(2, cfg.gmem_bytes.bit_length() - 1)]
        for i, a in enumerate(addrs):
            dev.write(a, (0xA5000000 + i).to_bytes(4, "little"))
        for i, a in enumerate(addrs):
            got = int.from_bytes(dev.read(a, 4), "little")
            if got != 0xA5000000 + i:
                raise AssertionError(f"address {a:#x} reads {got:#x}")
        return f"{n:,} random bytes and {len(addrs)} address lines"
    step("global memory", "check the serial link; if it fails at a fixed address, the bitstream may be wrong")(memory)

    def vecadd():
        n = 1000
        a = rng.integers(-2**31, 2**31, n, dtype=np.int64).astype(np.int32)
        b = rng.integers(-2**31, 2**31, n, dtype=np.int64).astype(np.int32)
        dev.reset_heap()
        pa, pb, pc = dev.upload(a), dev.upload(b), dev.alloc(n * 4)
        dev.load(assemble_file(str(KERNELS / "vecadd.s")))
        run = dev.launch(grid=(n + cfg.max_threads - 1) // cfg.max_threads, block=cfg.max_threads,
                         args=[pa, pb, pc, n])
        if not (dev.download(pc, n) == (a.astype(np.int64) + b).astype(np.int32)).all():
            raise AssertionError("wrong sums")
        return "1,000 additions" + (f" in {run.cycles:,} cycles" if run.cycles else "")
    step("first kernel (vector add)", "instruction memory or kernel launch; compare with the emulator")(vecadd)

    def reduction():
        dev.reset_heap()
        grid = 8
        x = rng.integers(-10**6, 10**6, grid * cfg.max_threads, dtype=np.int64).astype(np.int32)
        px, py = dev.upload(x), dev.alloc(grid * 4)
        dev.load(assemble_file(str(KERNELS / "reduce_sum.s")))
        dev.launch(grid=grid, block=cfg.max_threads, args=[px, py])
        if not (dev.download(py, grid) == x.reshape(grid, -1).sum(axis=1)).all():
            raise AssertionError("wrong block sums")
        return "shared memory, barriers and divergence"
    step("tree reduction", "shared memory or barrier logic")(reduction)

    def network():
        from ml import mnist
        from ml.gpu import GpuMLP
        from ml.qmlp import QuantMLP, preprocess

        dev.reset_heap()
        model = QuantMLP.load()
        data = mnist.load("test")
        gpu = GpuMLP(dev, model)
        correct = cycles = 0
        for i in range(images):
            q0 = preprocess(data["test_images"][i])
            res = gpu.run(q0)
            if not np.array_equal(res.hidden, model.hidden(q0)):
                raise AssertionError(f"image {i}: hidden activations differ from the NumPy reference")
            if not np.array_equal(res.logits, model.logits(q0)):
                raise AssertionError(f"image {i}: output scores differ from the NumPy reference")
            correct += res.prediction == data["test_labels"][i]
            cycles += sum(r.cycles or 0 for r in res.runs)
            show_digit(dev, res.prediction)
        per = f", {cycles // images:,} GPU cycles per digit" if cycles else ""
        return f"{images} test images bit-exact, {correct}/{images} recognized{per}"
    if images > 0:
        step("digit network", "a hardware fault the simpler tests missed: report the image index")(network)


def show_digit(dev: Device, value: int) -> None:
    """Show a 4-bit value on the board's LEDs 3-6 (FPGA builds only)."""
    if isinstance(dev.backend, BusBackend) and not isinstance(dev.backend, EmuBackend):
        dev.backend.bus_write(isa.HOST_BOARD_BASE, [value & 0xF])


# ------------------------------------------------------------- board handling


def check_bitstream(bitstream: Path, revision: str = fpga_build.DEFAULT_REVISION,
                    board: str = "ulx3s-85f") -> dict:
    """Accept only a completed build of these sources for the selected board (and PCB)."""
    if not bitstream.exists():
        raise FileNotFoundError(f"{bitstream} not found: build it with {build_command(board, revision)}")
    manifest = bitstream.with_name("summary.json")
    if not manifest.is_file():
        raise ValueError("build summary.json is missing; rebuild the bitstream")
    info = json.loads(manifest.read_text())
    if info.get("status") != "ready" or info.get("timing_passed") is not True:
        raise ValueError("build did not complete with passing timing; rebuild before programming")
    if board == "basys3":
        if info.get("target") != "basys3":
            raise ValueError(f"bitstream is for {info.get('target')}, expected basys3")
    elif info.get("target") != "ulx3s-85f" or info.get("pcb_revision") != revision:
        raise ValueError(f"bitstream is for {info.get('target')} PCB {info.get('pcb_revision')}, "
                         f"expected ulx3s-85f PCB {revision}")
    if info.get("bitstream_sha256") != fpga_build.sha256(bitstream):
        raise ValueError("bitstream checksum differs from its build summary; rebuild")
    if not info.get("sources"):
        raise ValueError("build summary has no source checksums; rebuild")
    changed = [name for name, digest in info["sources"].items()
               if not (ROOT / name).is_file() or fpga_build.source_sha256(ROOT / name) != digest]
    if changed:
        raise ValueError(f"bitstream is stale; changed sources: {', '.join(changed)}; rebuild")
    return info


def select_port(ports, requested: str | None = None, board: str = "ulx3s-85f"):
    if requested:
        matches = [p for p in ports if p.device.casefold() == requested.casefold()]
        if not matches:
            raise RuntimeError(f"serial port {requested} was not found; use --list")
    else:
        b = BOARDS[board]
        matches = [p for p in ports if (p.vid, p.pid) == b["usb"]]
        if not matches:
            raise RuntimeError(f"no {b['chip']} USB-serial device found")
        if board == "basys3" and len(matches) > 1:
            # Linux lists both FTDI channels; the serial port is the second (interface 1)
            matches = [p for p in matches if (p.location or "").endswith(".1")] or matches
    if len(matches) != 1:
        raise RuntimeError(f"several found ({', '.join(p.device for p in matches)}): pass --port")
    return matches[0]


def run_programmer(args: list[str], *, timeout: float | None = None) -> str:
    name = "fujprog" if os.name == "nt" else "openFPGALoader"
    fpga_build.find_tools((name,))
    p = subprocess.run([shutil.which(name), *args], capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        if p.returncode & 0xFFFFFFFF == 0xC0000135:
            raise RuntimeError(f"{name} cannot load a Windows runtime DLL. Install the FTDI CDM/D2XX "
                               "driver (ftd2xx.dll); see docs/fpga.md")
        raise RuntimeError(f"{name} exited with {p.returncode}: " + (p.stdout + p.stderr).strip()[-800:])
    return (p.stdout + p.stderr).strip()


def program(bitstream: Path, serial_number: str | None = None, board: str = "ulx3s-85f") -> None:
    """Load volatile FPGA SRAM; do not overwrite the factory flash image."""
    if board == "basys3":
        vivado_build.program(bitstream, BOARDS[board]["part"], serial_number)
        return
    if os.name == "nt":
        args = ["-S", serial_number] if serial_number else []
    else:
        args = ["-b", "ulx3s"] + (["--ftdi-serial", serial_number] if serial_number else [])
    run_programmer([*args, str(bitstream)])


def doctor(bitstream: Path, revision: str, board: str = "ulx3s-85f") -> int:
    """Report every software prerequisite without opening USB or serial devices."""
    def build_tools():
        fpga_build.find_tools()
        return "yosys, nextpnr-ecp5, ecppack found"

    def vivado():
        version = subprocess.run([vivado_build.find_vivado(), "-version"],
                                 capture_output=True, text=True, timeout=120)
        return (version.stdout + version.stderr).strip().splitlines()[0]

    def serial_library():
        import serial
        return f"pyserial {serial.VERSION}"

    if board == "basys3":
        tools = [("Vivado (build and programmer)", "install Vivado; see docs/fpga.md", vivado)]
    else:
        tools = [("build tools", "install the OSS CAD Suite; see docs/fpga.md", build_tools),
                 ("programmer", "install the FTDI CDM/D2XX driver on Windows; see docs/fpga.md",
                  lambda: run_programmer(["-V"], timeout=10).splitlines()[0])]
    checks = [
        *tools,
        ("serial library", 'python -m pip install -e ".[fpga]"', serial_library),
        ("build", build_command(board, revision),
         lambda: str(check_bitstream(bitstream, revision, board)["bitstream"])),
    ]
    failed = False
    for name, hint, check in checks:
        try:
            step(name, hint)(check)
        except StepFailed:
            failed = True
    if not failed:
        print(f"software checks passed; connect the board to {BOARDS[board]['connector']} "
              "to run the hardware checks")
    return int(failed)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", choices=list(BOARDS), default="ulx3s-85f")
    ap.add_argument("--port", help="serial port (default: the only board of that kind found)")
    ap.add_argument("--baudrate", type=int, default=115200)
    ap.add_argument("--bitstream", type=Path, help="default: the board's build (for --revision on a ULX3S)")
    ap.add_argument("--revision", choices=fpga_build.REVISIONS, default=fpga_build.DEFAULT_REVISION,
                    help="ULX3S PCB 3.1.7 (also 3.1.6), or 3.0 (also v2.x)")
    ap.add_argument("--check-build", action="store_true", help="verify the bitstream without connecting a board")
    ap.add_argument("--doctor", action="store_true", help="check tools, programmer, Python and bitstream without a board")
    ap.add_argument("--no-program", action="store_true", help="the board is already programmed")
    ap.add_argument("--images", type=int, default=20)
    ap.add_argument("--list", action="store_true", help="list serial ports and exit")
    ap.add_argument("--emulator", action="store_true", help="run the checks on the emulator")
    a = ap.parse_args(argv)
    if a.images < 0 or a.images > 10000 or a.baudrate <= 0:
        ap.error("--images must be 0..10000 and --baudrate must be positive")
    board = BOARDS[a.board]
    bitstream = a.bitstream or default_bitstream(a.board, a.revision)
    rebuild = "rebuild with " + build_command(a.board, a.revision)

    if a.doctor:
        return doctor(bitstream, a.revision, a.board)

    if a.check_build:
        try:
            step("build", rebuild)(lambda: str(check_bitstream(bitstream, a.revision, a.board)["bitstream"]))
        except StepFailed:
            return 1
        return 0

    if a.emulator:
        try:
            selftest(Device(EmuBackend(Config(gmem_bytes=64 * 1024))), images=a.images)
        except StepFailed:
            return 1
        print("all checks passed (emulator)")
        return 0

    try:
        from serial.tools import list_ports
    except ImportError:
        print('pyserial is missing: python -m pip install -e ".[fpga]"')
        return 1
    if a.list:
        ports = list(list_ports.comports())
        for p in ports:
            tag = "".join(f"  <- {b['name']}" for b in BOARDS.values() if (p.vid, p.pid) == b["usb"])
            print(f"{p.device:12s} {p.description}{tag}")
        if not ports:
            print(f"No serial ports found. Connect the board to {board['connector']} with a USB data cable.")
        return 0

    try:
        port = None
        ports = list(list_ports.comports())

        def usb():
            nonlocal port
            port = select_port(ports, a.port, a.board)
            if not a.no_program:
                if (port.vid, port.pid) != board["usb"]:
                    raise RuntimeError(f"selected port is not an {board['chip']}; "
                                       "use --no-program for a custom serial link")
                found = [p for p in ports if (p.vid, p.pid) == board["usb"]]
                if a.board == "basys3":   # one board can list two FTDI channels
                    found = [p for p in found if p.serial_number != port.serial_number] + [port]
                if len(found) > 1 and (not port.serial_number or
                        sum(p.serial_number == port.serial_number for p in found) != 1):
                    raise RuntimeError("boards do not have unique USB serial numbers; connect only the intended board")
            return f"using {port.device} ({port.serial_number or 'no USB serial number'})"
        step("USB connection", f"use a data cable (not charge-only) on {board['connector']}; on Windows the "
             "FTDI driver installs from Windows Update; try `python fpga/bringup.py --list`")(usb)

        if not a.no_program:
            def verify_build():
                info = check_bitstream(bitstream, a.revision, a.board)
                if info["params"]["BAUD"] != a.baudrate:
                    raise ValueError("--baudrate differs from the bitstream; use " + str(info["params"]["BAUD"]))
                which = "Basys 3" if a.board == "basys3" else f"85F PCB {a.revision}"
                return f"{which}, timing passed, source and bitstream checksums match"
            step("build", rebuild)(verify_build)
            load_hint = ("Vivado's hardware manager needs its cable drivers (part of the Vivado install); "
                         "check that the board is switched on" if a.board == "basys3" else
                         "on Windows fujprog needs the standard FTDI driver; close other programs "
                         "using the port; check that the board is an 85F")
            step("load the bitstream", load_hint)(lambda: program(bitstream, port.serial_number, a.board))
            time.sleep(1.0)

        from rosenblatt.uart import UartBackend
        holder = {}

        def connect():
            holder["backend"] = UartBackend(port.device, baudrate=a.baudrate)
            return f"serial protocol v1, bursts of {holder['backend'].max_words} words"
        step("serial link", f"press {board['reset']} to reset the GPU and retry; the baud rate must match "
             "the bitstream (115200 unless you changed top.v)")(connect)
        with holder["backend"] as backend:
            selftest(Device(backend), images=a.images)
    except StepFailed:
        return 1
    print("all checks passed: the GPU works on the board")
    return 0


if __name__ == "__main__":
    sys.exit(main())
