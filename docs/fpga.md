# Running Rosenblatt on an FPGA

The supported bring-up target is the **ULX3S ECP5-85F**, including Mouser
[CS-ULX3S-03](https://www.mouser.ca/en/ProductDetail/Radiona/CS-ULX3S-03).
The listing specifies **LFE5U-85F-6BG381C**: speed grade 6, CABGA381 package.
The build uses that exact device, its 25 MHz oscillator, and 115,200-baud 8N1
serial communication through **US1**. No external programmer is needed.

The board is not yet available for physical testing. Simulation and routed
timing checks are the evidence available before it arrives. This build uses
64 KiB of internal block RAM for global memory; it does **not** use the
board's external SDRAM. The complete digit network fits in that 64 KiB.

The open-source flow is Yosys (synthesis), nextpnr (place and route), ecppack
(bitstream), and fujprog on Windows or openFPGALoader on Linux/macOS.

## Which board

The design targets the [ULX3S](https://radiona.org/ulx3s/), an open-hardware
board with a Lattice ECP5 FPGA, an FTDI USB-serial chip, 8 LEDs and buttons.
It comes in four sizes; the build has a GPU configuration for each of the
three that can hold the digit network:

| Target | FPGA | GPU configuration |
| --- | --- | --- |
| `ulx3s-85f` | LFE5U-85F | 2 cores x 8 lanes x 4 warps, 32-bit multiply, DP4A (the default design) |
| `ulx3s-45f` | LFE5U-45F | 1 core x 8 lanes x 4 warps |
| `ulx3s-25f` | LFE5U-25F | 1 core x 4 lanes x 4 warps, 16-bit multiply, DP4A |

Every variant keeps the whole digit network (51 KB of weights) in 64 KiB of
on-chip block RAM that serves as global memory. The smallest ULX3S (12F) has
only 72 KB of block RAM in total, which is not enough for that plus the
register files and shared memory.

The 25F and 45F configurations are retained for experiments; the validation
below is for the 85F. Do not reuse earlier timing figures after changing RTL.
The completed build's `summary.json` records final routed timing, resource
use, tool versions, source checksums, and the bitstream checksum.

## Choose the PCB revision

The Mouser part number identifies the FPGA size but does not guarantee the
PCB revision. Read the version printed on the board when it arrives:

| Printed PCB version | Build / bring-up option | Bitstream folder under `build/fpga/` |
| --- | --- | --- |
| v3.1.6 or v3.1.7 | `--revision 3.1.7` (default) | `ulx3s-85f/` |
| v2.x or v3.0.x | `--revision 3.0` | `ulx3s-85f+pcb3.0/` |

Clock, serial, button and LED pins are common to these revisions. The ESP32
boot pin, held high by the design, moved from L2 to F1. The build appends the
correct revision-specific constraints to `pins.lpf`. Other prototype
revisions need their pinout checked before programming. These mappings come
from the board's [v20 constraints](https://github.com/emard/ulx3s/blob/master/doc/constraints/ulx3s_v20.lpf),
[v316 constraints](https://github.com/emard/ulx3s/blob/master/doc/constraints/prototype/ulx3s_v316.lpf),
and [revision history](https://github.com/emard/ulx3s/blob/master/doc/MANUAL.md#board-versions).

## Build

From the repository root, using the Python environment in the README:

```bash
python -m pip install -e ".[fpga,sim]" pytest
python fpga/build.py ulx3s-85f                  # PCB v3.1.6/v3.1.7
python fpga/build.py ulx3s-85f --revision 3.0   # older PCB pinout
python fpga/build.py --list
```

Install the [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build).
The scripts find it on `PATH`, at `~/tools/oss-cad-suite`, or at the directory
in `OSS_CAD_SUITE`. On Windows, use `.\.venv\Scripts\python.exe` for `python`
if the environment is not activated. No tool installation is performed by
the scripts.

Routing can take a substantial amount of time. The default is `router1`,
placement seed 1, and speed grade 6. `nextpnr.log` records progress. A build
must meet the **25 MHz constraint after routing** before `top.bit` is
produced. A failed, interrupted, or synthesis-only rebuild invalidates any
previous `top.bit`. If sources change during a build, it must be rerun.

Check the result without a board attached:

```bash
python fpga/bringup.py --check-build
python fpga/bringup.py --check-build --revision 3.0
python fpga/bringup.py --doctor
python fpga/bringup.py --emulator --images 20
```

`--check-build` verifies passing timing, PCB revision, and source/bitstream
checksums. `report.json` is nextpnr's full report. `summary.json` identifies
the exact sources that produced the image. Build products remain under the
ignored `build/` directory; keep a matching summary beside a copied image.

`--doctor` checks build tools, starts the programmer in version-only mode,
checks pyserial, and validates the bitstream. It reports every failure in
one run and never opens a USB or serial device. This catches a missing
Windows FTDI runtime before attempting to program a board. Append
`--revision 3.0` when checking an older PCB's image.

## Program the board

Plug the ULX3S in with a USB *data* cable (charge-only cables are common and do
not work) on the port marked US1. Both tools below are in the OSS CAD Suite.

On **Windows**, use `fujprog`, the ULX3S's own programmer. It works with the
standard FTDI driver that Windows installs automatically, so the board's
serial port keeps working for the demo:

The standard [FTDI CDM/D2XX driver](https://ftdichip.com/drivers/d2xx-drivers/)
provides both the serial port and the `ftd2xx.dll` runtime needed by fujprog.
If fujprog exits immediately with `0xC0000135`, install that driver; having
the programmer executable alone is not enough. See the programmer's
[Windows notes](https://github.com/kost/fujprog#troubleshooting).

For the 64-bit OSS CAD Suite, a missing runtime can also be supplied locally:
download FTDI's Windows Desktop driver ZIP, then copy
`amd64/FTD2XX64.dll` beside `fujprog.exe` as `ftd2xx.dll`. Use the 64-bit file,
not the one in `i386/`. `fujprog -V` and `python fpga/bringup.py --doctor`
verify that it loads. The USB driver is still needed when the board is
connected; Windows Update normally installs it at that point.

```bash
fujprog build/fpga/ulx3s-85f/top.bit          # volatile SRAM; lost at power-off
```

On **Linux and macOS**, `openFPGALoader` is the usual choice:

```bash
openFPGALoader -b ulx3s build/fpga/ulx3s-85f/top.bit       # into SRAM
```

(On Windows `openFPGALoader` needs the FTDI interface switched to the WinUSB
driver with Zadig, which hides the serial port; `fujprog` avoids that.) On
Linux, add yourself to the `dialout` group (or install the board's udev rule)
so you can open the serial port without root.

## First contact: the bring-up script

```bash
python -m pip install -e ".[fpga]"            # pyserial
python fpga/bringup.py --list                 # identify the serial port
python fpga/bringup.py --port COM5            # PCB v3.1.6/v3.1.7
# For v2.x/v3.0.x, append --revision 3.0 to the previous command.
```

Use the revision printed on the PCB. If only one matching FT231X is present,
`--port` may be omitted. With multiple boards, the selected serial port's
USB serial number is passed to the programmer so that programming and
testing address the same board. Duplicate USB serial numbers require
disconnecting the other boards first.

The script verifies the build, loads **volatile SRAM**, and checks each layer,
stopping at the first failure with a hint. It leaves the factory flash image
intact, so a power cycle restores the previous configuration:

1. USB connection (cable, port US1, driver)
2. checking the build and loading its bitstream
3. the serial link (protocol handshake)
4. the hardware configuration reported by the GPU
5. global memory (all 64 KiB of random data and every address line)
6. a first kernel (vector add)
7. shared memory, barriers and divergence (a tree reduction)
8. the digit network on 20 test images, every hidden activation and output
   score compared with the NumPy reference

`--no-program` skips step 2 if the board is already programmed, `--list`
shows the serial ports this PC sees, and `--emulator` runs steps 4-8 on the
emulator with the board's 64 KiB memory limit. The MNIST test split downloads
on first use if it is missing. `--images 0` runs the initial hardware checks
without a dataset download.

## Run the digit demo

The board appears as a serial port (Windows: `COM3`, `COM5`, ... in Device
Manager; Linux: `/dev/ttyUSB0`; macOS: `/dev/cu.usbserial-*`).

```bash
python -m pip install -e ".[fpga]"            # pyserial
python -m ml.demo --port COM5 --count 20
```

The demo asks the board for its configuration, uploads the network, then
sends test images one at a time. For each image it compares all 64 hidden
activations and all 10 output scores with the NumPy reference before printing
the prediction, so a wrong answer from the hardware cannot go unnoticed. It
also reports the GPU cycles spent per image (read from the board's counters).

To draw digits yourself: `python -m ml.draw --port COM5`.
The drawing demo checks both network layers before showing a prediction.
If communication or verification fails, it clears the prediction, displays
the error, and stops inference. Press FIRE1 and restart the demo to reconnect.

LEDs: **0** kernel running, **1** data arriving from the PC, **2** data going
back, **3-6** the recognized digit in binary (LED 3 is the lowest bit; the
demos write it to the board register at host address `0x300000`), **7**
heartbeat. Press **FIRE1** to reset the GPU; reset stays asserted until about
2.62 ms after the button settles. A reset is required
after any serial error (`docs/serial.md` explains why the host never retries).

## Benchmark the board

After a successful bring-up, close the drawing demo and run:

```bash
python bench/board.py --port COM5 --count 32 --output build/bench/my-board.json
```

This compares one image per inference with the shared-memory tiled kernel on
the same programmed GPU and the same MNIST images. Both strategies compare
all hidden activations and logits with NumPy. The benchmark replaces the
device's model and scratch buffers; it leaves the bitstream running.

The tiled runtime selects the largest group of eight that fits the device's
remaining global memory. On the 64 KiB 85F build, the trained model and its
padded biases occupy 52,608 bytes. Eight input/output slots add 7,808 bytes,
for **60,416 bytes total**. A 32-image run therefore reuses one eight-image
buffer four times, with the weights uploaded once for each strategy.

`BatchMLP(device, model)` selects this capacity automatically;
`gpu.run_batches(inputs)` yields one result per chunk.
The full input is validated before the first inference write. In the
benchmark, each result is checked before the next chunk starts.

Options:

- Omit `--port` to check the same workload on a 64 KiB emulator.
- `--start 100 --count 33` selects a different range and exercises a partial
  final tile. Each tile executes eight slots, so this run executes 40 slots
  but reports throughput per **33 real images**. Padding is zeroed; padded
  predictions are discarded. Choose a multiple of eight for steady batches.
- `--scalar` disables DP4A in both strategies.
- `--batch 8` sets the tiled capacity explicitly. Values round up to a
  multiple of eight; a capacity that does not fit fails before the tiled
  model is uploaded.
- `--baudrate` must match the loaded image. `--clock-hz` defaults to
  25,000,000 and is used only to convert GPU counters to seconds; it does
  not measure or change the board's clock.

The report separates three timings: model setup, host inference, and NumPy
verification. **GPU cycles** come from the device's counters and exclude
serial transfers. **Host inference time** includes input/program uploads,
launches, polling, counter reads, and output downloads. At 115,200 baud,
serial transfers can dominate the time you wait even when a kernel is fast.
The emulator records event counts and wall time, with GPU timing left null.

JSON reports include the device configuration, model/input/source checksums,
per-launch counters, actual and padded image counts, and predictions. The
default destination is `build/bench/board-latest.json`. An error or Ctrl+C
saves a failed or interrupted report, including completed chunks, and
replaces any previous success at that path. A mismatch reports the image
and neuron index; press FIRE1 before restarting after a serial failure.

The older profiling study in `bench/results/` used 256 KiB and 32 images per
tiled launch. Its 3,139 cycles per image is a simulation result for that
workload. Use this benchmark for the 64 KiB board's eight-image launches and
keep compute time separate from host time when comparing results.

## If it does not work

| Symptom | Likely cause |
| --- | --- |
| LED 7 not blinking | bitstream not loaded, or built for a different FPGA size |
| `serial response timed out` right away | wrong port, or the bitstream's baud rate differs from `--baudrate` (default 115,200) |
| `serial bridge rejected request: BUS_TIMEOUT` | the GPU stopped answering; press FIRE1 and reconnect |
| `unsupported GPU hardware version` | bitstream and Python code from different versions |
| demo stops with a mismatch | a real hardware bug: please report the image index and the build target |
| build is stale / checksum mismatch | rebuild from the current sources before programming |
| `fujprog` cannot load a Windows runtime DLL | install the FTDI CDM/D2XX driver; keep the COM-port driver |

## Verification before the board arrives

```bash
python -m pytest -m "not rtl" -q
python -m pytest tests/rtl -q
```

`tests/rtl/test_ulx3s.py` instantiates the real board wrapper with two cores,
eight lanes, four warps, 64 KiB global memory and the 25 MHz clock. One test
uses the real 115,200-baud host timing. The full-network test accelerates only
serial transfer; it checks reduction, single-image inference and tiled
inference against the emulator, including hardware event counters. Both
tests verify power-on reset, serial activity LEDs, the displayed digit,
resetting a running kernel during a partial packet, and reconnection.
The broader suite covers corrupt packets, stalled bus transfers, arithmetic,
divergence, memory and multiple GPU configurations.

Passing these tests and routing does not replace programming a physical
board. The final acceptance step is a successful bring-up run on the chosen
PCB revision.

`python fpga/record_validation.py` writes the validation record
(`fpga/validation/ulx3s-85f.json`) after both images are built and the test
runs listed in its docstring have produced their reports. It refuses to write
anything if an image is stale, a report failed, or a report is older than the
sources.

### Recorded results: 2026-10-12

These results are for the Verilog as of commit `49a2209`. The changes after
it (word-only store merging, the register scoreboard, `dp4ab`, the matrix
unit and its slots) changed the sources, so rebuild before programming a
board; `bringup.py` refuses a stale image.

Both 85F builds completed with speed grade 6, `router1`, and placement seed 1:

| PCB profile | Final routed maximum clock | Required clock | Result |
| --- | ---: | ---: | --- |
| v3.1.6/v3.1.7 (`3.1.7`) | 32.71 MHz | 25 MHz | PASS |
| v2.x/v3.0.x (`3.0`) | 37.02 MHz | 25 MHz | PASS |

Both use 44,382 / 83,640 logic cells (53%), 8,366 flip-flops, 100 / 208 block
RAMs and 112 / 156 DSP multipliers. They still run at the board's **25 MHz**;
the timing results do not change the UART divider or operating clock.

Validation passed 105 fast tests, eight GPU/UART RTL tests, and both board
wrapper tests (physical baud and full-network inference). The emulator
bring-up checked all 64 KiB, vector addition, reduction, and 20 MNIST images;
every hidden activation and output score matched, with 20/20 predictions
correct. The drawing demo also passed its hidden-window check and rejected
injected serial and hidden-layer failures. Windows `--doctor` passed for
both images after supplying fujprog's FTDI runtime.

The [validation record](../fpga/validation/ulx3s-85f.json) contains the exact
source checksums, tool versions, bitstream checksums and test results. These
are simulation and software-tool results; physical-board testing is pending.

## Another board

`fpga/ulx3s/top.v` contains a synchronized button/power-on reset, the GPU with its UART
bridge (`rtl/rb_serial_gpu.v`, board independent), and LED drivers. For
another ECP5 board, copy `fpga/ulx3s/`, change the pin names in the `.lpf`
file and the clock frequency in `top.v`, and add a target in
`fpga/build.py`. For a Lattice iCE40 board, use `synth_ice40` and
`nextpnr-ice40` instead. The iCE40 UP5K (5,280 LUTs) is too small for even
the 4-lane configuration (13,872 logic cells on the ECP5); an iCE40 build
would need something like 2 lanes and a much smaller shared-memory crossbar.

## What takes the space

Earlier synthesis of one 8-lane core (Yosys `synth_ecp5 -noflatten`, then
summed per block). This breakdown explains the architecture; use the current
build summary for the final device totals:

| Block | LUT4 | Carry cells | Flip-flops | DSP | Block RAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 ALUs | 7,952 | 640 | 1,440 | 56 | |
| shared memory: 8 banks and the lane/bank crossbar | 6,203 | | 82 | | 8 |
| pipeline, warp scheduler, divergence stacks, counters | 2,706 | 576 | 1,144 | | 2 |
| load/store unit (coalescer, 4 instructions in flight) | 1,933 | 6 | 458 | | |
| top level: control registers, dispatcher | 1,500 | 80 | 516 | | |
| 8 register files (3 read copies each) | 968 | | 616 | | 24 |
| global memory (64 KiB) and arbiter | 70 | 16 | 273 | | 32 |

Lessons from getting here:

- **Crossbars are expensive.** The shared memory's lane-to-bank routing is
  29% of the core, as much as six ALUs. The first version was 14,600 LUTs. Its
  variable-index assignments (`x[bank*32 +: 32] = ...`) synthesized into
  full-width shifters; rewritten as one-hot AND-OR logic, and with stores
  served one lane per bank, it shrank to 6,200.
- **In a wide memory, stores are the expensive direction.** With 32-byte
  global-memory lines (`--param GMEM_LINE=32`), each lane picks its word out
  of a loaded line, an 8-to-1 multiplexer per lane. Merging stores into a line
  needs every byte to choose among all lanes, with the highest lane winning;
  that made the whole GPU 40% larger. Stores now coalesce within one word
  only, and the 32-byte build costs 11% more LUTs than the 4-byte one
  (39,312 against 35,550 in synthesis), for a 1.9x faster one-image kernel
  when the weights are interleaved (see the write-up, section 4).
- **Watch where a shared multiplexer goes.** `dp4ab` needs one 8-to-1, 32-bit
  multiplexer per core. Wired to the ALU's general operand input, it grew the
  GPU by 21%: to keep paths short, synthesis copied it into every lane's
  adders, shifters and comparators. Wired only to the dot-product multipliers,
  it costs 4%.
- **DSP blocks run out before logic does.** Each lane needs 3 DSPs for a
  32x32 multiply and 4 for DP4A. The 25F has 28 DSPs, so its 4-lane build
  uses the 16-bit multiplier (1 DSP) and keeps DP4A.
- **Borrowing hardware costs multiplexers.** The matrix unit
  (`--param ENABLE_MMA=1`) has no multipliers of its own: on the 85F, 112 of
  the 156 DSPs are already in use, and another four per lane would not fit.
  It borrows each lane's DP4A multipliers through a 2-to-1 multiplexer on
  every multiplier input, and keeps the accumulator tiles and the warps'
  operand slots in LUT RAM. Synthesis: 39,013 LUTs against 35,432 (10% more),
  9,294 flip-flops against 8,380, 956 LUT RAM blocks against 440, and the
  same 112 DSPs. Routed (PCB v3.1.7, `router1`, seed 1), it uses 50,689 of
  83,640 logic cells (61%) and reaches 36.42 MHz against the 25 MHz required.
  With the parameter off, the unit leaves nothing behind.
- **A second slot is nearly free.** `--param MMA_SLOTS=2` lets each warp
  have two operations in the matrix unit. Their operands fit in the 16-word
  LUT RAM blocks the first slot already uses, so it adds 48 flip-flops, no
  LUT RAM and no DSPs. Its LUT cost is too small to separate from synthesis
  noise: Yosys 0.70 maps the one-slot unit to 39,069 LUTs and the two-slot
  unit to 39,361, but it mapped the previous commit's one-slot unit, the same
  logic written differently, to 39,476. Routed with the OSS CAD Suite of
  2026-10-19 (nextpnr 0.11.1-65, PCB v3.1.7, `router1`, seed 1), the two-slot
  build uses 51,047 of 83,640 logic cells (61%) and reaches 30.58 MHz. Its
  critical path does not touch the matrix unit: it runs from a register file
  through the branch logic to the divergence stack, and 72% of it is
  routing.
- **Simulation-only code must stay out of synthesis.** Loops that zero the
  memories at time 0 are fine in a simulator. Yosys unrolls them element by
  element (65,536 iterations for the default global memory) and appeared to
  hang. They are wrapped in `` `ifndef SYNTHESIS ``; FPGA block RAM powers up as
  zero anyway.
- **The longest path ran through the multiplier.** It went from a register
  file read, through a DSP multiplier and long wires, to the adder tree of
  `dp4a`/`mad`. The ALU now registers the partial products and finishes the
  sum in the write-back stage. That shortens the path without costing a cycle,
  because results are written back in that stage anyway.
