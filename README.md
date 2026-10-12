# Rosenblatt

A tiny GPU designed from scratch, and a neural network that runs on it.

- **Hardware**: a SIMT GPU in Verilog: warps of 8 threads in lockstep, 4
  warps per core interleaved to hide memory latency, a divergence stack,
  banked shared memory, coalescing load/store units, and an int8
  dot-product instruction (DP4A). The default build has 2 cores x 8 lanes x 4
  warps, 256 KiB of global memory and 4 KiB of shared memory per core.
- **Software**: an instruction set, an assembler, a reference emulator, and a
  Python runtime that drives the emulator, the simulated Verilog, or an FPGA
  over a serial link with the same calls.
- **Network**: a 784-64-10 MLP trained in PyTorch and quantized to int8. It
  recognizes 97.80% of the MNIST test digits; every layer, including rounding,
  ReLU and saturation, runs as GPU instructions.
- **FPGA**: targets the ULX3S ECP5-85F at 25 MHz, with 64 KiB of internal
  global memory and a UART connection to the PC. The build checks routed
  timing and records source/bitstream checksums. Both common PCB pinouts
  are supported; physical-board testing is still pending.
- **Profiling and comparison**: a cycle-accurate profiling study of the
  network on the GPU, and the same network on an NVIDIA RTX 5080 for
  comparison.

**Read [docs/writeup.md](docs/writeup.md)** for the design decisions, the
measurements, and why the RTX 5080 is so much faster.

## Run the digit demo

From the repository root, with Python 3.10 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m ml.demo --count 10
```

On Linux/macOS, use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`.

The saved model in `ml/artifacts/mlp_int8.npz` needs only NumPy. The demo
downloads the MNIST test split into `data/mnist` if it is missing (checking
checksums), runs each image on the emulator, and checks all 64 hidden
activations and all 10 output scores against an independent NumPy reference
before printing the prediction.

```powershell
.\.venv\Scripts\python.exe -m ml.demo --start 100 --count 20   # other images
.\.venv\Scripts\python.exe -m ml.demo --count 10 --scalar      # without DP4A
.\.venv\Scripts\python.exe -m ml.demo --port COM5              # on a ULX3S board
```

Or draw your own digits with the mouse; the GPU recognizes them as you
release the button and shows all ten scores and the 28x28 image it saw:

```powershell
.\.venv\Scripts\python.exe -m ml.draw                 # emulator
.\.venv\Scripts\python.exe -m ml.draw --port COM5     # ULX3S; its LEDs show the answer too
```

The FPGA path is described in [docs/fpga.md](docs/fpga.md); `python
fpga/bringup.py` takes a new board from "plugged in" to "recognizing digits"
one checked step at a time.

To prepare an 85F before it arrives:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[fpga]"
.\.venv\Scripts\python.exe fpga/build.py ulx3s-85f
.\.venv\Scripts\python.exe fpga/bringup.py --doctor
```

The default pinout is PCB v3.1.6/v3.1.7. For v2.x/v3.0.x, add
`--revision 3.0` to the build and bring-up commands. Check the printed board
revision before programming; the [FPGA guide](docs/fpga.md) covers US1,
Windows drivers, reset, and the first hardware test. `--doctor` checks the
software and bitstream without opening any USB or serial device.

After bring-up, compare single-image inference with batches sized for the
board's memory:

```powershell
.\.venv\Scripts\python.exe bench/board.py --port COM5 --count 32
.\.venv\Scripts\python.exe bench/board.py --count 32          # emulator check before the board arrives
```

The tiled path streams eight images at a time through 64 KiB. Both paths
verify every hidden activation and output score and save a JSON report in
`build/bench/board-latest.json`. Hardware counters measure GPU compute;
host time includes serial transfers. The emulator reports no GPU timing.
See [the benchmark guide](docs/fpga.md#benchmark-the-board) for options and
how to interpret partial batches.

## Results

Cycle-accurate simulation of the default GPU with 256 KiB of global memory,
using real MNIST test images (32 images per tiled batch):

| Strategy | Cycles per digit | Global memory requests per digit |
| --- | ---: | ---: |
| One image at a time (`kernels/dense_int8.s`), scalar | 70,324 | 57,316 |
| One image at a time, DP4A | 17,796 | 14,404 |
| Batched, shared-memory tiles (`kernels/dense_tiled.s`), DP4A | 3,139 | 2,100 |

Moving less data (shared memory) gained 5.7x and doing four multiplies per
instruction (DP4A) gained 4x. The one-image kernel is limited by memory
bandwidth, not arithmetic.

Optional memory-system upgrades go further. A wider memory port
(`GMEM_LINE=32`: 32 bytes per request instead of 4) makes the one-image kernel
1.9x faster, but only when the weights are stored so that neighbouring threads
read neighbouring words (`layout="interleaved"`); with row-major weights it
changes almost nothing. Adding a register scoreboard (`SCOREBOARD=1`: a load
stalls only the instructions that need its register), a loop unrolled by two,
and an 8-instruction load queue brings the one-image kernel to 6,057 cycles,
2.9x faster than the baseline. These hardware options cost 11% more logic and
are off in the board build. A cross-lane dot product, `dp4ab` (one lane's
operand broadcast to the whole warp, 4% more logic), cuts the tiled kernel's
instructions by 37%; with the memory options it runs in 1,690 cycles, 1.86x
faster than the baseline.

The same network on an RTX 5080 Laptop GPU (`cuda/`), 10,000 images, every
output score identical to the reference:

| Method | ns per digit |
| --- | ---: |
| naive CUDA | 123.5 |
| DP4A | 32.0 |
| shared-memory tiles + DP4A | 7.2 |
| cuBLAS int8 (tensor cores) | 4.1 |

At the FPGA's 25 MHz clock, the simulated 3,139 cycles correspond to 125.6
microseconds per digit, 17,500 times slower than the RTX 5080 running the same
algorithm. This estimate excludes serial transfers and uses a larger batch
than fits the 64 KiB board; `bench/board.py` measures the board-sized workload.
The write-up breaks down the estimate: 113x clock, 480x lanes, and utilization
working in the other direction.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[sim]" pytest
.\.venv\Scripts\python.exe -m pytest -m "not rtl" -q                             # fast, no simulator
.\.venv\Scripts\python.exe -m pytest -q                                            # everything
```

RTL tests need Icarus Verilog (`iverilog`, `vvp`) on `PATH`. The harness also
finds an OSS CAD Suite installation at `~/tools/oss-cad-suite` or at
`$OSS_CAD_SUITE`.

Every scenario runs on the emulator and on the Verilog in four hardware
configurations, and outputs *and* performance counters must agree exactly.
Scenarios: vector addition, naive and tiled matrix multiply, a tree reduction,
random ALU programs, random nested divergence with loops and early returns,
every load/store flavour with coalescing and bank conflicts, and three neural
network paths. The UART bridge is tested at the pin level, and the whole GPU
runs the trained network through the simulated serial pins. The ULX3S board
wrapper is also tested for startup, real serial timing, LED outputs, and
reset/reconnection while a kernel and a partial serial packet are in flight.

## Source map

| Path | Purpose |
| --- | --- |
| `rtl/` | GPU cores, ALUs, register files, shared and global memory, UART bridge |
| `sw/rosenblatt/` | ISA, assembler, emulator, runtime, serial driver |
| `kernels/` | assembly kernels |
| `ml/` | training, quantization, NumPy reference, GPU inference (`gpu.py`), demo |
| `tests/` | emulator tests, cocotb RTL tests, UART tests |
| `bench/` | profiling study on the RTL (`profile_mnist.py`, `report.py`, `results/`) |
| `fpga/` | ULX3S top level, pin constraints, build script |
| `cuda/` | the same network on an NVIDIA GPU |
| `docs/` | [write-up](docs/writeup.md), [architecture](docs/architecture.md), [ISA](docs/isa.md), [FPGA](docs/fpga.md), [serial protocol](docs/serial.md) |

To retrain, install the `ml` extra and run `python ml/train.py` (about 10
seconds on a laptop CPU). Training downloads the training split and replaces
the artifacts in `ml/artifacts`.
