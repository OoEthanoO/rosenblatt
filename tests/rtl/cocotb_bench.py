"""cocotb module for the profiling study (bench/profile_mnist.py drives it).

Runs one MNIST inference strategy on the simulated GPU, checks every logit
against the NumPy reference, and writes the cycle counts and performance
counters to the JSON file named by $RB_OUT.

    RB_WORKLOAD = per_image | tiled      RB_DP4A = 0 | 1      RB_COUNT = images
    RB_LAYOUT = rows | interleaved (weight layout, default rows)
    RB_UNROLL = 1 | 2 (one-image kernel only, default 1)
"""

from __future__ import annotations

import json
import os

import cocotb
import numpy as np
from cocotb.task import bridge

from cocotb_gpu import CocotbBackend, config_from_env, reset
from rosenblatt.device import Device


def run_bench(dut) -> dict:
    from ml import mnist
    from ml.gpu import BatchMLP, GpuMLP
    from ml.qmlp import QuantMLP, preprocess

    cfg = config_from_env()
    workload = os.environ["RB_WORKLOAD"]
    dp4a = os.environ.get("RB_DP4A", "1") == "1"
    count = int(os.environ["RB_COUNT"])
    layout = os.environ.get("RB_LAYOUT", "rows")
    unroll = int(os.environ.get("RB_UNROLL", "1"))
    model = QuantMLP.load()
    data = mnist.load("test")
    q0 = preprocess(data["test_images"][:count])
    dev = Device(CocotbBackend(dut, cfg))

    runs = []
    if workload == "per_image":
        gpu = GpuMLP(dev, model, dp4a=dp4a, layout=layout, unroll=unroll)
        logits = []
        for i in range(count):
            res = gpu.run(q0[i])
            logits.append(res.logits)
            runs += res.runs
        logits = np.array(logits)
    else:
        res = BatchMLP(dev, model, batch=count, dp4a=dp4a, layout=layout).run(q0)
        logits, runs = res.logits, res.runs
    np.testing.assert_array_equal(logits, model.logits(q0))
    correct = int((np.argmax(logits, axis=1) == data["test_labels"][:count]).sum())

    totals: dict[str, int] = {}
    for r in runs:
        for core in r.per_core:
            for k, v in core.items():
                totals[k] = totals.get(k, 0) + v
    return {
        "workload": workload, "dp4a": dp4a, "layout": layout, "unroll": unroll, "count": count,
        "correct": correct,
        "config": json.loads(os.environ["RB_CONFIG"]),
        "cycles": sum(r.cycles for r in runs),
        "launch_cycles": [r.cycles for r in runs],
        "device_gmem_reqs": sum(r.gmem_reqs for r in runs),
        "counters": totals,
    }


@cocotb.test(timeout_time=500, timeout_unit="ms")   # 50M cycles: a hang fails instead of spinning
async def bench(dut):
    await reset(dut)
    result = await bridge(run_bench)(dut)
    with open(os.environ["RB_OUT"], "w") as f:
        json.dump(result, f, indent=2)
    dut._log.info("%s dp4a=%s: %d images, %d cycles (%.0f per image), %d/%d correct",
                  result["workload"], result["dp4a"], result["count"], result["cycles"],
                  result["cycles"] / result["count"], result["correct"], result["count"])
