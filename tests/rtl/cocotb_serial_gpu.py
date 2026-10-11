"""Use the production Python driver with the simulated serial GPU."""

import os

import cocotb
from cocotb.task import bridge

from uart_sim import BlockingSerial, SerialPins, reset
from cocotb_gpu import compare, config_from_env
from rosenblatt.device import Device, EmuBackend
from rosenblatt.uart import UartBackend
from scenarios import SCENARIOS


def run_serial(dut, pins):
    cfg = config_from_env()
    with UartBackend(stream=BlockingSerial(pins), timeout=120, kernel_timeout=120) as backend:
        assert backend.cfg == cfg
        assert backend.max_words == 64
        dev = Device(backend)
        base = cfg.gmem_bytes - 32
        dev.write(base, b"Z" * 32)
        dev.write(base + 3, bytes(range(23)))
        assert dev.read(base, 32) == b"Z" * 3 + bytes(range(23)) + b"Z" * 6
        for name in os.environ["RB_SCENARIOS"].split(","):
            expected = SCENARIOS[name](Device(EmuBackend(cfg)), 0)
            dev.reset_heap()
            actual = SCENARIOS[name](dev, 0)
            assert not (errors := compare(name, expected, actual)), "\n".join(errors)
            assert all(run.cycles > 0 for run in actual.runs)
            dut._log.info("UART %s: outputs and counters match; %d GPU cycles", name,
                          sum(run.cycles for run in actual.runs))


@cocotb.test(timeout_time=250, timeout_unit="ms")
async def serial_gpu(dut):
    await reset(dut)
    pins = SerialPins(dut)
    await bridge(run_serial)(dut, pins)
