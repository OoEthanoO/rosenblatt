"""Build failure handling and first-use checks, without FPGA tools or hardware."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from fpga import build, bringup


@pytest.mark.parametrize("report,ok", [
    ({}, False),
    ({"fmax": {"clk": {"achieved": 24.9, "constraint": 25}}}, False),
    ({"fmax": {"clk": {"achieved": 30, "constraint": 20}}}, False),
    ({"fmax": {"clk": {"achieved": 25, "constraint": 25}}}, True),
])
def test_timing_is_required(report, ok):
    assert build.timing_passes(report, 25) is ok


@pytest.fixture
def fake_build(tmp_path, monkeypatch):
    real_root = build.ROOT
    for path in [*(build.RTL / f for f in build.RTL_FILES), build.RTL / "rb_isa.vh",
                 real_root / "fpga/build.py", *sorted((real_root / "fpga/ulx3s").glob("*"))]:
        target = tmp_path / path.relative_to(real_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(build, "RTL", tmp_path / "rtl")
    monkeypatch.setattr(bringup, "ROOT", tmp_path)
    monkeypatch.setattr(build, "BOARD", {**build.BOARD, "top": tmp_path / "fpga/ulx3s/top.v",
                                         "lpf": tmp_path / "fpga/ulx3s/ulx3s.lpf"})
    monkeypatch.setattr(build.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="test tool", stderr=""))
    control = {"timing": 30, "fail": False, "edit": False}

    def run(cmd, log):
        if control["fail"]:
            raise RuntimeError("tool failed")
        out = log.parent
        if cmd[0] == "yosys":
            (out / "synth_stat.txt").write_text(" 100 DP16KD\n")
        elif cmd[0] == "nextpnr-ecp5":
            assert "--timing-allow-fail" not in cmd
            assert cmd[cmd.index("--speed") + 1] == "6"
            progress = json.loads((out / "summary.json").read_text())
            assert progress["status"] == "routing"
            assert progress["cells"]["DP16KD"] == 100
            (out / "report.json").write_text(json.dumps({
                "fmax": {"clk": {"achieved": control["timing"], "constraint": 25}}}))
            if control["edit"]:
                with (build.RTL / "rb_gpu.v").open("a") as f:
                    f.write("\n// source changed during build\n")
        elif cmd[0] == "ecppack":
            Path(cmd[-1]).write_bytes(b"test bitstream")
        return 0.01

    monkeypatch.setattr(build, "run", run)
    return control


@pytest.mark.parametrize("revision,pin", [("3.0", "L2"), ("3.1.7", "F1")])
def test_build_matches_revision_and_detects_stale_artifacts(fake_build, revision, pin):
    info = build.build("ulx3s-85f", revision=revision)
    bit = build.ROOT / info["bitstream"]
    assert bringup.check_bitstream(bit, revision)["status"] == "ready"
    assert f'LOCATE COMP "wifi_gpio0" SITE "{pin}"' in bit.with_name("pins.lpf").read_text()
    other = "3.0" if revision == "3.1.7" else "3.1.7"
    with pytest.raises(ValueError, match="expected"):
        bringup.check_bitstream(bit, other)
    bit.write_bytes(b"damaged bitstream")
    with pytest.raises(ValueError, match="checksum"):
        bringup.check_bitstream(bit, revision)
    bit.write_bytes(b"test bitstream")
    (build.RTL / "rb_gpu.v").write_text("changed")
    with pytest.raises(ValueError, match="stale"):
        bringup.check_bitstream(bit, revision)


@pytest.mark.parametrize("failure", ["timing", "fail", "edit"])
def test_failed_rebuild_cannot_leave_an_old_bitstream(fake_build, failure):
    info = build.build("ulx3s-85f")
    bit = build.ROOT / info["bitstream"]
    fake_build[failure] = 20 if failure == "timing" else True
    with pytest.raises(RuntimeError):
        build.build("ulx3s-85f")
    assert not bit.exists()
    assert json.loads(bit.with_name("summary.json").read_text())["status"] == "failed"


def test_synthesis_only_invalidates_an_older_programming_image(fake_build):
    info = build.build("ulx3s-85f")
    bit = build.ROOT / info["bitstream"]
    bit.with_name("top.config").write_text("old routed design")
    assert build.build("ulx3s-85f", synth_only=True)["status"] == "synthesized"
    assert not bit.exists()
    assert not bit.with_name("top.config").exists()
    assert not bit.with_name("report.json").exists()
    assert json.loads(bit.with_name("summary.json").read_text())["finished_utc"]


@pytest.mark.parametrize("params", [{"CLK_HZ": 50_000_000}, {"BAUD": 0}, {"NOT_A_PARAMETER": 1}])
def test_invalid_board_overrides_fail_before_build(params):
    with pytest.raises(ValueError):
        build.build("ulx3s-85f", extra=params)


def port(name, serial):
    return SimpleNamespace(device=name, serial_number=serial, vid=0x0403, pid=0x6015)


def test_selecting_a_serial_port_is_unambiguous():
    ports = [port("COM3", "first"), port("COM5", "second")]
    with pytest.raises(RuntimeError, match="several"):
        bringup.select_port(ports)
    assert bringup.select_port(ports, "COM5").serial_number == "second"
    with pytest.raises(RuntimeError, match="not found"):
        bringup.select_port(ports, "COM9")


def test_programmer_selects_the_same_usb_device(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(bringup.fpga_build, "find_tools", lambda required: None)
    monkeypatch.setattr(bringup.shutil, "which", lambda name: name)
    monkeypatch.setattr(bringup.subprocess, "run", lambda cmd, **kw:
                        calls.append(cmd) or SimpleNamespace(returncode=0, stdout="", stderr=""))
    bringup.program(tmp_path / "top.bit", "second")
    assert "second" in calls[0]
    assert not any(arg in calls[0] for arg in ("flash", "-f"))


def test_missing_windows_programmer_runtime_has_an_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setattr(bringup.fpga_build, "find_tools", lambda required: None)
    monkeypatch.setattr(bringup.shutil, "which", lambda name: name)
    monkeypatch.setattr(bringup.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(returncode=0xC0000135, stdout="", stderr=""))
    with pytest.raises(RuntimeError, match="FTDI CDM/D2XX"):
        bringup.program(tmp_path / "top.bit")


@pytest.mark.parametrize("missing_runtime", [False, True])
def test_doctor_checks_software_without_opening_hardware(fake_build, monkeypatch, capsys, missing_runtime):
    info = build.build("ulx3s-85f")
    monkeypatch.setattr(bringup.fpga_build, "find_tools", lambda *a: None)
    monkeypatch.setitem(sys.modules, "serial", SimpleNamespace(VERSION="test"))
    calls = []

    def version(args, **kw):
        calls.append((args, kw))
        if missing_runtime:
            raise RuntimeError("FTDI CDM/D2XX driver missing")
        return "programmer version test\n"

    monkeypatch.setattr(bringup, "run_programmer", version)
    assert bringup.main(["--doctor", "--bitstream", str(build.ROOT / info["bitstream"])]) == int(missing_runtime)
    assert calls == [(["-V"], {"timeout": 10})]
    output = capsys.readouterr().out
    assert "PASS  serial library" in output and "PASS  build (" in output
    if missing_runtime:
        assert "FAIL  programmer" in output
        assert "software checks passed" not in output
