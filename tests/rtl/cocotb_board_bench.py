"""Run the production benchmark through the 85F wrapper and real UART driver."""

import json
import os
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.task import bridge
from cocotb.triggers import ClockCycles, FallingEdge

from uart_sim import BlockingSerial, SerialPins
from cocotb_gpu import MATCHED_STATS, config_from_env
from bench.board import run_benchmark
from ml.qmlp import QuantMLP
from rosenblatt.device import Device, EmuBackend
from rosenblatt.uart import UartBackend


def compare_benchmarks(pins):
    model = QuantMLP.load()
    inputs = np.random.default_rng(81).integers(0, 128, (9, 784), dtype=np.int32)
    labels = np.argmax(model.logits(inputs), axis=1)
    cfg = config_from_env()
    expected = run_benchmark(Device(EmuBackend(cfg)), model, inputs, labels)
    with UartBackend(stream=BlockingSerial(pins), timeout=180, kernel_timeout=180) as backend:
        assert backend.cfg == cfg
        actual = run_benchmark(Device(backend), model, inputs, labels, clock_hz=25_000_000,
                               progress=lambda message: print(message, flush=True))
    assert actual["config"] == expected["config"]
    for ref, got in zip(expected["strategies"], actual["strategies"]):
        assert got["verified_images"] == got["correct"] == 9
        assert got["allocated_bytes"] == ref["allocated_bytes"]
        assert got["kernel_cycles"] > 0
        assert got["kernel_seconds"] == got["kernel_cycles"] / 25_000_000
        assert len(got["chunks"]) == len(ref["chunks"])
        for ref_chunk, got_chunk in zip(ref["chunks"], got["chunks"]):
            assert ref_chunk["predictions"] == got_chunk["predictions"]
            assert ref_chunk["padded_count"] == got_chunk["padded_count"]
            assert len(ref_chunk["launches"]) == len(got_chunk["launches"]) == 2
            for ref_run, got_run in zip(ref_chunk["launches"], got_chunk["launches"]):
                for key in MATCHED_STATS:
                    assert ref_run["stats"].get(key, 0) == got_run["stats"].get(key, 0), key
    tiled = actual["strategies"][1]
    assert tiled["batch_capacity"] == 8 and tiled["allocated_bytes"] == 60_416
    assert [c["count"] for c in tiled["chunks"]] == [8, 1]
    assert tiled["padded_images"] == 16
    assert actual["kernel_speedup"] > 1
    # Preserve the detailed result for inspection without mistaking simulator
    # wall time for the speed of the physical serial link.
    actual.update(backend="rtl_serial", status="passed", input_kind="seeded random (seed 81)",
                  physical_board_tested=False, baudrate=6_250_000,
                  note="UART accelerated for simulation; host wall times are simulator execution times.")
    output = Path(os.environ["RB_BENCH_OUT"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(actual, indent=2) + "\n")
    return actual


@cocotb.test(timeout_time=2, timeout_unit="sec")
async def benchmark(dut):
    dut.btn.value = 0
    dut.ftdi_txd.value = 1
    cocotb.start_soon(Clock(dut.clk_25mhz, 40, unit="ns").start())
    await ClockCycles(dut.clk_25mhz, 65540)
    await FallingEdge(dut.clk_25mhz)
    assert dut.rst.value == 0
    pins = SerialPins(dut, clk=dut.clk_25mhz, rx=dut.ftdi_txd,
                      tx=dut.ftdi_rxd, period_ns=40, baudrate=6_250_000)
    result = await bridge(compare_benchmarks)(pins)
    dut._log.info("64 KiB benchmark: both layers and all event counters match, tiled speedup %.2fx",
                  result["kernel_speedup"])
