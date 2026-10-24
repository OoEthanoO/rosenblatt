"""The Vivado flow for the Basys 3, and bring-up's board handling, without Vivado or hardware."""

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from fpga import build as ecp5_build, bringup, vivado_build

# report_utilization tables as Vivado prints them (after synthesis the LUT rows carry an asterisk)
UTILIZATION = """
1. Slice Logic
--------------

+-------------------------+-------+-------+------------+-----------+-------+
|        Site Type        |  Used | Fixed | Prohibited | Available | Util% |
+-------------------------+-------+-------+------------+-----------+-------+
| Slice LUTs{star}             | 12345 |     0 |          0 |     20800 | 59.35 |
|   LUT as Logic          | 10257 |     0 |          0 |     20800 | 49.31 |
|   LUT as Memory         |  2088 |     0 |          0 |      9600 | 21.75 |
| Slice Registers         |  4136 |     0 |          0 |     41600 |  9.94 |
+-------------------------+-------+-------+------------+-----------+-------+

3. Memory
---------

+-------------------+------+-------+------------+-----------+-------+
|     Site Type     | Used | Fixed | Prohibited | Available | Util% |
+-------------------+------+-------+------------+-----------+-------+
| Block RAM Tile    | 16.5 |     0 |          0 |        50 | 33.00 |
+-------------------+------+-------+------------+-----------+-------+

4. DSP
------

+----------------+------+-------+------------+-----------+-------+
|    Site Type   | Used | Fixed | Prohibited | Available | Util% |
+----------------+------+-------+------------+-----------+-------+
| DSPs           |   56 |     0 |          0 |        90 | 62.22 |
| DSP48E1 only   |   56 |       |            |           |       |
+----------------+------+-------+------------+-----------+-------+
"""


def test_utilization_report_is_parsed():
    for star in ("", "*"):
        u = vivado_build.parse_utilization(UTILIZATION.format(star=star))
        assert u == {"luts": {"used": 12345, "available": 20800},
                     "luts_as_memory": {"used": 2088, "available": 9600},
                     "registers": {"used": 4136, "available": 41600},
                     "bram_tiles": {"used": 16.5, "available": 50},
                     "dsps": {"used": 56, "available": 90}}


def test_timing_needs_every_clock_to_pass():
    clocks = vivado_build.parse_clocks("sys_clk_pin 10.000 none none\nclk_mmcm 40.000 15.200 0.051\n")
    assert clocks["clk_mmcm"] == {"period_ns": 40.0, "setup_slack_ns": 15.2, "hold_slack_ns": 0.051}
    assert vivado_build.timing_passes(clocks)
    assert not vivado_build.timing_passes(vivado_build.parse_clocks("clk_mmcm 40.000 -0.100 0.051\n"))
    assert not vivado_build.timing_passes(vivado_build.parse_clocks("clk_mmcm 40.000 1.000 -0.010\n"))
    assert not vivado_build.timing_passes(vivado_build.parse_clocks("sys_clk_pin 10.000 none none\n"))


@pytest.fixture
def fake_vivado(tmp_path, monkeypatch):
    """A copy of the Basys 3 sources, and a Vivado that writes reports from `control`."""
    real_root = vivado_build.ROOT
    for path in [*(vivado_build.RTL / f for f in ecp5_build.RTL_FILES), vivado_build.RTL / "rb_isa.vh",
                 real_root / "fpga/vivado_build.py", *sorted((real_root / "fpga/basys3").glob("*"))]:
        target = tmp_path / path.relative_to(real_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    monkeypatch.setattr(vivado_build, "ROOT", tmp_path)
    monkeypatch.setattr(vivado_build, "RTL", tmp_path / "rtl")
    monkeypatch.setitem(vivado_build.TARGETS, "basys3", {
        **vivado_build.TARGETS["basys3"], "top": tmp_path / "fpga/basys3/top.v",
        "xdc": tmp_path / "fpga/basys3/basys3.xdc"})
    monkeypatch.setattr(bringup, "ROOT", tmp_path)
    monkeypatch.setattr(vivado_build, "find_vivado", lambda: "vivado")
    monkeypatch.setattr(vivado_build.subprocess, "run",
                        lambda *a, **kw: SimpleNamespace(returncode=0, stdout="vivado v2025.1 (64-bit)\n", stderr=""))
    control = {"setup_slack": 15.2, "fail": False, "edit": False, "scripts": []}

    def run(tcl, log):
        script = tcl.read_text()
        control["scripts"].append(script)
        if control["fail"]:
            raise RuntimeError("vivado exited with 1")
        out = log.parent
        (out / "utilization_synth.txt").write_text(UTILIZATION.format(star="*"))
        if "route_design" in script:
            (out / "utilization.txt").write_text(UTILIZATION.format(star=""))
            (out / "clocks.txt").write_text(f"sys_clk_pin 10.000 none none\n"
                                            f"clk_mmcm 40.000 {control['setup_slack']:.3f} 0.051\n")
            if control["setup_slack"] >= 0:   # build.tcl writes no bitstream when timing fails
                (out / "top.bit").write_bytes(b"test bitstream")
            if control["edit"]:
                with (vivado_build.RTL / "rb_gpu.v").open("a") as f:
                    f.write("\n// source changed during build\n")
        return 0.01

    monkeypatch.setattr(vivado_build, "run_vivado", run)
    return control


def test_build_writes_a_checked_summary(fake_vivado):
    info = vivado_build.build("basys3")
    assert info["status"] == "ready" and info["timing_passed"] is True
    assert info["fmax_mhz"] == {"clk_mmcm": round(1000 / (40 - 15.2), 2)}
    assert info["utilization"]["dsps"] == {"used": 56, "available": 90}
    assert info["params"]["NUM_CORES"] == 1 and info["clock_mhz"] == 25
    script = fake_vivado["scripts"][0]
    assert "-part xc7a35tcpg236-1" in script and "-verilog_define SYNTHESIS=1" in script
    assert "basys3.xdc" in script and "write_bitstream" in script
    bit = vivado_build.ROOT / info["bitstream"]
    assert bringup.check_bitstream(bit, board="basys3")["status"] == "ready"
    with pytest.raises(ValueError, match="expected"):
        bringup.check_bitstream(bit)             # not a ULX3S image
    bit.write_bytes(b"damaged bitstream")
    with pytest.raises(ValueError, match="checksum"):
        bringup.check_bitstream(bit, board="basys3")
    bit.write_bytes(b"test bitstream")
    top = vivado_build.ROOT / "fpga/basys3/top.v"
    top.write_text(top.read_text() + "// changed\n")
    with pytest.raises(ValueError, match="stale"):
        bringup.check_bitstream(bit, board="basys3")


def test_parameters_pass_to_synthesis(fake_vivado):
    info = vivado_build.build("basys3", extra={"SCOREBOARD": 1})
    assert "-generic SCOREBOARD=1" in fake_vivado["scripts"][0]
    assert info["bitstream"].startswith("build/fpga/basys3+SCOREBOARD=1/")


@pytest.mark.parametrize("failure", ["timing", "fail", "edit"])
def test_failed_vivado_build_cannot_leave_an_old_bitstream(fake_vivado, failure):
    info = vivado_build.build("basys3")
    bit = vivado_build.ROOT / info["bitstream"]
    if failure == "timing":
        fake_vivado["setup_slack"] = -0.4
    else:
        fake_vivado[failure] = True
    with pytest.raises(RuntimeError):
        vivado_build.build("basys3")
    assert not bit.exists()
    summary = json.loads(bit.with_name("summary.json").read_text())
    assert summary["status"] == "failed" and summary["finished_utc"]


def test_synthesis_only_counts_resources_and_invalidates_the_image(fake_vivado):
    info = vivado_build.build("basys3")
    bit = vivado_build.ROOT / info["bitstream"]
    summary = vivado_build.build("basys3", synth_only=True)
    assert summary["status"] == "synthesized" and summary["utilization"]["luts"]["used"] == 12345
    assert "route_design" not in fake_vivado["scripts"][-1]
    assert not bit.exists()


@pytest.mark.parametrize("params", [{"CLK_HZ": 30_000_000}, {"BAUD": 0}, {"NOT_A_PARAMETER": 1}])
def test_invalid_overrides_fail_before_vivado_runs(fake_vivado, params):
    with pytest.raises(ValueError):
        vivado_build.build("basys3", extra=params)
    assert fake_vivado["scripts"] == []


def port(name, serial, pid=0x6010, location=None):
    return SimpleNamespace(device=name, serial_number=serial, vid=0x0403, pid=pid, location=location)


def test_basys3_serial_port_is_found_on_windows_and_linux():
    ulx3s = port("COM3", "ulx3s", pid=0x6015)
    assert bringup.select_port([ulx3s, port("COM4", "210183ABCDEFB")], board="basys3").device == "COM4"
    # Linux lists the JTAG channel too; the serial port is interface 1
    linux = [port("/dev/ttyUSB0", "210183ABCDEF", location="1-1:1.0"),
             port("/dev/ttyUSB1", "210183ABCDEF", location="1-1:1.1")]
    assert bringup.select_port(linux, board="basys3").device == "/dev/ttyUSB1"
    with pytest.raises(RuntimeError, match="FT2232H"):
        bringup.select_port([ulx3s], board="basys3")
    assert bringup.select_port([ulx3s, port("COM4", "x")]).device == "COM3"   # the ULX3S is still the default


def test_basys3_is_programmed_through_vivado(fake_vivado, tmp_path):
    bit = tmp_path / "top.bit"
    bit.write_bytes(b"test bitstream")
    bringup.program(bit, "210183ABCDEFB", "basys3")
    script = fake_vivado["scripts"][-1]
    assert "{*210183ABCDEF*}" in script                  # the board's serial without the channel letter
    assert "get_hw_devices xc7a35t*" in script and bit.as_posix() in script
    assert "program_hw_devices" in script and "flash" not in script.lower()


def test_basys3_doctor_checks_vivado_without_hardware(fake_vivado, monkeypatch, capsys):
    info = vivado_build.build("basys3")
    monkeypatch.setitem(sys.modules, "serial", SimpleNamespace(VERSION="test"))
    bit = vivado_build.ROOT / info["bitstream"]
    assert bringup.main(["--board", "basys3", "--doctor", "--bitstream", str(bit)]) == 0
    output = capsys.readouterr().out
    assert "PASS  Vivado (build and programmer)" in output and "vivado v2025.1" in output
    assert "PASS  build (" in output and "PROG/UART" in output
    shutil.rmtree(bit.parent)
    assert bringup.main(["--board", "basys3", "--doctor", "--bitstream", str(bit)]) == 1
    assert "python fpga/vivado_build.py basys3" in capsys.readouterr().out
