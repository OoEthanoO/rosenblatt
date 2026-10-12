"""Record a pre-hardware validation run of the ULX3S-85F images.

    python fpga/record_validation.py

Run it after both routed images are built and the test runs below have
written their JUnit reports:

    python fpga/build.py ulx3s-85f
    python fpga/build.py ulx3s-85f --revision 3.0
    python -m pytest -m "not rtl" --junitxml=build/fast-tests.xml
    python -m pytest tests/rtl/test_rtl.py tests/rtl/test_serial_rtl.py --junitxml=build/rtl-regression-tests.xml
    python -m pytest tests/rtl/test_ulx3s.py --junitxml=build/ulx3s-tests.xml
    python -m pytest tests/rtl/test_board_bench_rtl.py --junitxml=build/board-bench-tests.xml

It records nothing unless both images match the current sources and every
report passed and is newer than every source file. It then repeats the
bring-up checks on the emulator and writes fpga/validation/ulx3s-85f.json.
"""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "sw"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fpga import bringup, build  # noqa: E402
from rosenblatt.device import Device, EmuBackend  # noqa: E402
from rosenblatt.emu import Config  # noqa: E402

REPORTS = {
    "assembler, emulator, runtime, serial driver, build and preflight": "build/fast-tests.xml",
    "GPU configurations, UART faults and serial inference": "build/rtl-regression-tests.xml",
    "85F wrapper: physical baud, full network, reset recovery": "build/ulx3s-tests.xml",
    "85F wrapper: board benchmark through the UART pins": "build/board-bench-tests.xml",
}
SOURCE_DIRS = ["rtl", "sw", "tests", "kernels", "ml", "fpga/ulx3s"]


def newest_source() -> float:
    files = [f for d in SOURCE_DIRS for f in (ROOT / d).rglob("*")
             if f.is_file() and "__pycache__" not in f.parts]
    return max(f.stat().st_mtime for f in files)


def report(name: str, relative: str, sources_mtime: float) -> dict:
    path = ROOT / relative
    if not path.is_file():
        raise SystemExit(f"{relative} is missing: run the test command in this script's docstring")
    if path.stat().st_mtime < sources_mtime:
        raise SystemExit(f"{relative} is older than the sources: run its tests again")
    root = ET.parse(path).getroot()
    suites = list(root.iter("testsuite"))
    counts = {key: sum(int(s.get(key, 0)) for s in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    if counts["tests"] == 0 or counts["failures"] or counts["errors"]:
        raise SystemExit(f"{relative}: {counts}")
    return {"name": name, "result": "passed", **counts, "report": relative,
            "report_sha256": build.sha256(path), "started": suites[0].get("timestamp"),
            "seconds": round(sum(float(s.get("time", 0)) for s in suites), 3),
            "cases": [case.get("name") for case in root.iter("testcase")]}


def emulator_bringup(images: int = 20) -> dict:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        bringup.selftest(Device(EmuBackend(Config(gmem_bytes=64 * 1024))), images=images)
    lines = [line.strip() for line in out.getvalue().splitlines() if line.strip()]
    return {"command": f"python fpga/bringup.py --emulator --images {images}",
            "result": "passed", "output": lines}


def main() -> int:
    images = {}
    for revision in build.REVISIONS:
        try:
            images[revision] = bringup.check_bitstream(
                build.output_dir("ulx3s-85f", revision=revision) / "top.bit", revision)
        except (OSError, ValueError) as e:
            raise SystemExit(f"PCB {revision}: {e}") from None
    sources_mtime = newest_source()
    tests = [report(name, path, sources_mtime) for name, path in REPORTS.items()]
    try:
        programmer = bringup.run_programmer(["-V"], timeout=10).splitlines()[0]
    except Exception as e:  # noqa: BLE001 - recorded, not fatal
        programmer = f"unavailable: {e}"
    ftdi = ROOT / "build" / "tools" / "ftdi-runtime" / "source.json"
    record = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "board": "ULX3S ECP5-85F (LFE5U-85F-6BG381C)",
        "physical_board_tested": False,
        "builds": images,
        "tests": tests,
        "emulator_bringup": emulator_bringup(),
        "programmer_version": programmer,
        "ftdi_runtime": ({k: v for k, v in json.loads(ftdi.read_text()).items() if k != "installed_as"}
                         if ftdi.is_file() else None),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "uncommitted_changes": bool(subprocess.check_output(["git", "status", "--porcelain"],
                                                            cwd=ROOT, text=True).strip()),
    }
    destination = ROOT / "fpga" / "validation" / "ulx3s-85f.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {destination.relative_to(ROOT)}")
    for revision, info in images.items():
        print(f"  PCB {revision}: {info['fmax_mhz']} MHz, timing passed")
    for t in tests:
        print(f"  {t['tests']:4d} passed  {t['name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
