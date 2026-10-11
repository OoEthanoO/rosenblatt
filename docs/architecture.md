# Architecture

Rosenblatt is a SIMT GPU in the style of NVIDIA's designs from 2010 to 2016:
many threads run one program, grouped into warps that execute in lockstep, and
a core interleaves warps to hide memory latency. Everything is parameterized;
the default build used for simulation has

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `NUM_CORES` | 2 | cores (SMs); each runs one thread block at a time |
| `NUM_LANES` | 8 | threads per warp = SIMD lanes per core = shared-memory banks |
| `NUM_WARPS` | 4 | warps per core, so up to 32 threads per block |
| `IMEM_AW` | 10 | 1024 instructions of program memory |
| `GMEM_AW` | 16 | 256 KiB of global memory (64 KiB on the FPGA builds) |
| `SMEM_AW` | 10 | 4 KiB of shared memory per core |
| `GMEM_LATENCY` | 8 | cycles from a global-memory request to its data |
| `GMEM_INTERVAL` | 1 | cycles between requests the memory accepts (bandwidth) |
| `LSU_DEPTH` | 4 | global-memory instructions in flight per core |
| `MUL_WIDTH` | 32 | 16 = only the low 16 bits of each operand are multiplied |
| `ENABLE_DP4A` | 1 | packed int8 dot-product instruction |

```
                       host bus (testbench, or the UART bridge on an FPGA)
                                        |
                  +---------------------+----------------------+
                  |  control registers, block dispatcher,     |
                  |  performance counters                      |
                  +-----+--------------------------+-----------+
                        |                          |
   +--------------------+---------+  +-------------+-----------------+
   | core 0                        |  | core 1                        |
   |  warp scheduler (4 warps)     |  |  ...                          |
   |  instruction memory           |  |                               |
   |  8 lanes: register file + ALU |  |                               |
   |  shared memory (8 banks)      |  |                               |
   |  load/store unit              |  |                               |
   +--------------------+----------+  +-------------+-----------------+
                        |                          |
                  +-----+--------------------------+-----------+
                  |  global memory: one 32-bit request/cycle, |
                  |  round-robin between cores and the host   |
                  +--------------------------------------------+
```

## Launching a kernel (`rtl/rb_gpu.v`)

The host writes the grid size, block size, entry address and up to eight
arguments into control registers and sets `CTRL.start`. The dispatcher hands
block 0 to the first idle core, block 1 to the next, and so on; when a core
finishes a block it gets the next one. The kernel is done when every block has
finished. The cycle count of the launch and per-core performance counters can
then be read back (`isa.CSR`, `isa.Perf`).

## A core (`rtl/rb_core.v`)

A core holds up to four warps of the current block. Each warp has its own
program counter, active-thread mask, exited-thread mask and divergence stack.
The pipeline has four stages:

| Stage | Work |
| --- | --- |
| S0 | the scheduler picks a ready warp, round robin; instruction fetch |
| S1 | decode; all lanes read `rs1`, `rs2` and `rd` from their register files |
| S2 | execute: ALUs, shared memory, branch/stack update, hand-off to the LSU |
| S3 | write back results and shared-memory load data |

A warp never has two instructions between S0 and S2, so there are no data
hazards to detect and no forwarding paths. Instead the scheduler fills the
pipeline with *other* warps: one warp can issue every third cycle, so three
ready warps keep the core issuing every cycle. This is fine-grained
multithreading, the GPU answer to latency: instead of making one thread fast,
keep many threads ready.

**Register files** (`rb_regfile.v`): each lane stores 16 registers for each of
the 4 warps. Three copies of the file provide three read ports (`rs1`, `rs2`,
and `rd`, which is the accumulator of `mad`/`dp4a` and the data of stores);
every write goes to all copies. Reads are synchronous, so the files map onto
FPGA block RAM or LUT RAM.

**ALUs** (`rb_alu.v`): one per lane, purely combinational: add/sub, logic,
shifts, compares, min/max, a 32-bit (or 16-bit) multiplier shared by `mul` and
`mad`, and the optional `dp4a` (four 8-bit multiplies and an adder tree).

**Divergence**: a branch evaluates its condition in every active lane. If
the lanes disagree, the warp continues with the fall-through lanes and pushes
`{target, taken lanes}`; `sync` later pops it and runs the deferred lanes.
`ssy` pushes a reconvergence point, and `ret` removes threads for good. A warp
whose active mask becomes empty pops its stack automatically ("unwinding")
or, with an empty stack, finishes.

**Barriers**: `bar` parks a warp. When every warp of the block is parked or
finished, all are released.

## Shared memory (`rtl/rb_smem.v`)

Eight banks (one per lane), word-interleaved: word `i` lives in bank `i % 8`.
A warp's shared-memory instruction is executed in stage S2:

- each bank serves the row requested by the lowest waiting lane;
- loads that want that same word ride along (broadcast);
- stores write one lane per bank per cycle, in lane order, so the highest
  lane's data lands last ("highest lane wins") without a byte-merging network;
- lanes that were not served are a *bank conflict*: the instruction is
  re-issued (replayed) for just those lanes.

Conflict-free accesses (every lane a different bank, or all the same word)
cost one issue slot, exactly like an add. The data comes back in S3.

The lane-to-bank crossbar costs about as much logic as four ALUs (see
`docs/fpga.md`). Real GPUs pay the same price: shared memory is fast because a
lot of silicon is spent on routing.

## Global memory and the load/store unit (`rtl/rb_lsu.v`, `rtl/rb_gmem.v`)

Global memory is one array shared by every core and the host, behind a
round-robin arbiter. It accepts one 32-bit request every `GMEM_INTERVAL`
cycles and returns read data `GMEM_LATENCY` cycles later, in order. On an FPGA
it is block RAM; the extra latency stages model the DRAM a real GPU uses.

Each core's load/store unit has two parts:

1. **Issue**: takes one warp instruction and sends its requests, one per
   cycle. Requests are *coalesced*: the lowest waiting lane picks a word, and
   every lane whose address falls in that word is served by the same request.
   Eight lanes reading eight consecutive bytes cost two requests; reading one
   shared address costs one (broadcast); eight scattered addresses cost eight.
2. **In flight**: up to `LSU_DEPTH` instructions whose requests have all gone
   out wait for their data. Because the issue part is free as soon as the last
   request has left, several loads overlap in the memory pipeline
   (*memory-level parallelism*). Completions write back in order through the
   register-file port, taking free S3 slots (the scheduler skips one issue slot
   when it has to make room).

A warp that issued a global load waits (`WS_MEM`) until the data is written
back. Meanwhile the other warps run. A memory instruction that finds the
issue part busy is replayed.

## Host interface

The 32-bit host bus is word addressed: global memory at `0x000000`,
instruction memory at `0x100000` (write only, broadcast to every core) and
control registers at `0x200000`. In simulation cocotb drives it directly; on
an FPGA `rb_uart_bridge.v` turns serial frames into bus transactions
(`docs/serial.md`). The Python `Device` class hides the difference:

```python
dev = Device(EmuBackend())           # or CocotbBackend(dut) in simulation, UartBackend("COM5")
x = dev.upload(np.arange(100, dtype=np.int32))
dev.load(assemble_file("kernels/vecadd.s"))
run = dev.launch(grid=4, block=32, args=[x, y, z, 100])
run.cycles, run.per_core[0]["gmem_reqs"]
```

## Performance counters

Per core: active cycles, committed instructions, active lanes, idle cycles
(and how many of those had a warp waiting on memory), replays, global-memory
instructions, requests and lanes (their ratio is the coalescing factor),
shared-memory instructions and wavefronts (their ratio shows bank conflicts),
LSU busy cycles, divergent branches and blocks. The emulator counts the same
architectural events, and the RTL tests require both to agree exactly.

## Verification

- `sw/rosenblatt/emu.py` is the golden model: architectural semantics with no
  timing.
- `tests/scenarios.py`: vector add, two matrix multiplies, special registers,
  random ALU programs, random nested divergence with loops and early returns
  (checked against a per-thread interpreter), every load/store flavour,
  barriers, and three neural-network paths.
- `tests/rtl/`: every scenario on the Verilog under three hardware
  configurations (outputs and event counters must match the emulator exactly),
  the UART bridge at the pin level, and the whole GPU driven through UART.
- `bench/profile_mnist.py` checks every logit of every image it profiles.
