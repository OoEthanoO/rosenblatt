# Rosenblatt

A tiny SIMT GPU in Verilog, with a Python assembler, functional emulator,
device runtime, and an int8 neural network. The default configuration has two
cores, eight lanes per warp, four warps per core, 256 KiB of global memory,
and 4 KiB of shared memory per core.

The digit classifier runs both dense layers and hidden-layer rounding,
ReLU, and saturation as GPU instructions. Weights and intermediate activations
stay in device memory. Python preprocesses pixels and chooses the largest
output logit. The same runtime drives the emulator and the simulated RTL;
running on a physical FPGA still needs a board integration and host transport.

## Run the digit demo

From the repository root, using Python 3.10 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m ml.demo --count 10
```

On Linux/macOS, use `.venv/bin/python` in place of
`.\.venv\Scripts\python.exe`.

The saved model in `ml/artifacts/mlp_int8.npz` needs only NumPy for inference.
The demo downloads the MNIST test split into `data/mnist` if it is missing,
checks the files' checksums, and reuses the cache on later runs. Each image's
64 hidden activations and all ten logits must exactly match the independent
NumPy reference before its prediction is printed.

```powershell
# Different test images
.\.venv\Scripts\python.exe -m ml.demo --start 100 --count 20

# The same model using scalar int8 multiply-adds
.\.venv\Scripts\python.exe -m ml.demo --count 10 --scalar
```

The first ten MNIST test images were all classified correctly in both modes.
For those ten images, the emulator counted:

| Kernel mode | Warp instructions | Global-memory word requests |
| --- | ---: | ---: |
| DP4A (four signed int8 products per instruction) | 114,710 | 144,040 |
| Scalar int8 | 450,710 | 573,160 |

These are architectural event counts, not cycle counts or measured FPGA
speedups. The demo reports accuracy for the selected sample only. The saved
training report in `ml/artifacts/accuracy.txt` records 97.80% accuracy for the
deployed int8 model on the full MNIST test set.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[sim]" pytest

# Fast checks: no simulator, PyTorch, or dataset download needed
.\.venv\Scripts\python.exe -m pytest tests/test_gpu_mlp.py -q

# Full suite, including the Verilog simulations
.\.venv\Scripts\python.exe -m pytest -q
```

RTL tests require Icarus Verilog (`iverilog` and `vvp`) on `PATH`. The harness
also detects an OSS CAD Suite installation at `~/tools/oss-cad-suite`, or at
the directory specified by `OSS_CAD_SUITE`.

Tests compare emulator and RTL outputs and architectural counters across
three hardware configurations. They cover vector addition, matrix multiply,
divergence, memory access, barriers, and neural inference. The inference
scenarios include a small network with padded dimensions and reused buffers,
and the full trained 784 -> 64 -> 10 model on deterministic inputs. The small
hardware configuration exercises the scalar fallback with a 16-bit multiplier.
Additional emulator tests cover rounding, saturation, tied predictions,
invalid inputs, and accumulator overflow rejection.

## Source map

| Path | Purpose |
| --- | --- |
| `rtl/` | GPU cores, ALUs, register files, memory, and host bus |
| `sw/rosenblatt/` | ISA, assembler, emulator, and device runtime |
| `kernels/` | Assembly kernels, including `dense_int8.s` |
| `ml/gpu.py` | Upload the network and launch both layers on any `Device` |
| `ml/qmlp.py` | Quantized model and independent NumPy reference |
| `ml/demo.py` | Verified MNIST inference demo |
| `ml/train.py` | PyTorch training and quantization |
| `tests/` | Emulator checks and cocotb RTL comparisons |

To retrain, install the `ml` extra and run `python ml/train.py`. Training
downloads the training split and replaces the artifacts in `ml/artifacts`.
