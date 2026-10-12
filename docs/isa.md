# Rosenblatt ISA

Fifty-one instructions, 32 bits each: sixteen ALU operations in register and
immediate forms, `dp4ab`, ten loads and stores, and nine control instructions. Every
thread has sixteen 32-bit registers (`r0` is always zero). The definitions live in one place,
`sw/rosenblatt/isa.py`; the assembler, the emulator and the Verilog
(`rtl/rb_isa.vh` is generated from it) all read them from there.

## Execution model (what a kernel sees)

A kernel launch runs a *grid* of `grid` blocks with `block` threads each, like
CUDA's `kernel<<<grid, block>>>()`. All threads run the same program; they tell
themselves apart through special registers (`%tid`, `%ctaid`, ...). Threads are
grouped in warps of `NUM_LANES` (8 by default) that execute in lockstep. A block
holds at most `NUM_LANES x NUM_WARPS` threads (32 by default) and runs on one
core; its threads share that core's shared memory and can wait for each other
with `bar`.

Kernel arguments are eight 32-bit values (`%arg0` .. `%arg7`) set by the host,
typically device addresses and sizes. Memory is byte addressed and little-endian.

## Encoding

```
 31    26 25  22 21  18 17  14 13             0
+--------+------+------+------+----------------+
| opcode |  rd  | rs1  | rs2  |    unused      |  R-type  rd = rs1 OP rs2
| opcode |  rd  | rs1  | rs2  | unused  | lane |  dp4ab (lane in bits 3..0)
| opcode |  rd  | rs1  |     imm18 (signed)    |  I-type  rd = rs1 OP imm
| opcode |  rd  |         imm22 (signed)       |  J-type  li / lui
+--------+------+------+------+----------------+
```

The top two opcode bits choose a group: `00` ALU with a register operand, `01`
the same ALU operation with an immediate, `10` memory, `11` control. The word
`0x00000000` is `add r0, r0, r0`, a natural `nop`.

## ALU (opcodes `0x00`-`0x0F` register form, `0x10`-`0x1F` immediate form)

| fn | register form | immediate form | Meaning |
| ---: | --- | --- | --- |
| 0 | `add rd, ra, rb` | `addi rd, ra, imm` | rd = ra + b |
| 1 | `sub` | `subi` | rd = ra - b |
| 2 | `mul` | `muli` | rd = low 32 bits of ra * b ¹ |
| 3 | `mad` | `madi` | rd = ra * b + rd ¹ |
| 4 | `and` | `andi` | |
| 5 | `or` | `ori` | |
| 6 | `xor` | `xori` | |
| 7 | `shl` | `shli` | shift left by b[4:0] |
| 8 | `shr` | `shri` | logical shift right |
| 9 | `sra` | `srai` | arithmetic shift right |
| 10 | `slt` | `slti` | rd = (ra < b) signed, 0 or 1 |
| 11 | `sltu` | `sltui` | unsigned compare |
| 12 | `seq` | `seqi` | rd = (ra == b) |
| 13 | `min` | `mini` | signed minimum |
| 14 | `max` | `maxi` | signed maximum (`relu rd, ra` = `max rd, ra, r0`) |
| 15 | `dp4a` | - | rd += sum of the four products of the signed bytes of ra and rb ² |

Immediates are 18-bit signed values (-131072 .. 131071), sign-extended to 32
bits for every operation.

¹ With `MUL_WIDTH = 16` (small FPGA builds) only the low 16 bits of each operand
are multiplied, as signed numbers, like NVIDIA's old `mul24`. Kernels that keep
multiply operands in -32768 .. 32767 behave identically on both.
² Optional (`ENABLE_DP4A`); an illegal instruction where it is disabled.
NVIDIA has had the same instruction (`__dp4a`) since Pascal.

### `dp4ab rd, ra, rb, lane` (opcode `0x3A`)

The same dot product as `dp4a`, but every lane takes `ra` from one lane of the
warp: rd += sum of the four products of the signed bytes of *lane `lane`'s*
`ra` and this lane's `rb`. The lane number is taken modulo the warp width. All
lanes read their registers before any of them writes, so `dp4ab r5, r5, r6, 2`
uses lane 2's old `r5` everywhere. Reading a lane whose thread does not exist
gives an undefined value. It comes with `ENABLE_DP4A`.

In CUDA terms it is `acc += __dp4a(__shfl_sync(mask, a, lane), b)` in one
instruction. Eight of them in a row, one per lane, multiply an 8x4 tile of
int8 values by a 4x8 tile across the warp: a matrix instruction spread over
eight issue slots. The tiled network kernel uses it to read each input word
from shared memory once per warp instead of once per image
(`BatchMLP(..., bcast=True)`).

## Memory (opcodes `0x20`-`0x2F`)

| Instruction | Meaning |
| --- | --- |
| `ldw rd, [ra + imm]` | load word from global memory (address rounded down to a multiple of 4) |
| `ldb rd, [ra + imm]` | load byte, sign-extended |
| `ldbu rd, [ra + imm]` | load byte, zero-extended |
| `stw [ra + imm], rd` | store word (the data register sits in the rd field) |
| `stb [ra + imm], rd` | store the low byte of rd |
| `lds`, `ldsb`, `ldsbu`, `sts`, `stsb` | the same, for the block's shared memory |

Addresses wrap around the size of the memory. When several lanes of one store
write the same byte, the highest lane wins. Opcode bits: `10 S W ss` with S =
shared, W = store, ss = 00 word, 01 signed byte, 10 unsigned byte.

## Control (opcodes `0x30`-`0x3F`)

| Instruction | Meaning |
| --- | --- |
| `li rd, imm22` | rd = sign-extended 22-bit value |
| `lui rd, imm22` | rd = imm22 << 10 |
| `s2r rd, %name` | read a special register (table below) |
| `bz ra, label` / `bnz ra, label` | per-thread branch if ra == 0 / != 0 (absolute target) |
| `ssy label` | push a reconvergence point |
| `sync` | pop: go to the next deferred group of threads, or reconverge |
| `bar` | wait until every thread of the block has arrived (`__syncthreads()`) |
| `ret` | the thread finishes |

### Special registers

| Name | CUDA | |
| --- | --- | --- |
| `%tid` | `threadIdx.x` | thread index in the block |
| `%ntid` | `blockDim.x` | threads per block |
| `%ctaid` | `blockIdx.x` | block index |
| `%nctaid` | `gridDim.x` | blocks in the grid |
| `%gtid` | `blockIdx.x * blockDim.x + threadIdx.x` | global thread index |
| `%laneid`, `%warpid` | | position within the warp / block |
| `%coreid`, `%clock` | | which core, and its cycle counter |
| `%arg0` .. `%arg7` | kernel parameters | |

### Divergence

When the threads of a warp disagree about a branch, the hardware runs the
threads that fall through first and pushes `{target, taken threads}` on a
per-warp stack (8 entries). `sync` pops that entry and switches to the deferred
threads; `ssy` pushes the point where both groups meet again. This is how
NVIDIA GPUs before Volta (Fermi, Kepler, Maxwell, Pascal) did it, with the same
instruction names. Correct results never depend on `ssy`/`sync`: without them
threads that split simply stay split until they `ret`. Reconvergence only saves
time. The assembler generates the right pattern from structured macros:

```
    if.nz r3            ; threads with r3 != 0 run the then-part
        ...
    else
        ...
    endif               ; everyone continues together here

    loop                ; do { ... } while (r5 != 0), per thread
        ...
    endloop.nz r5
```

## Assembler

```
python -m rosenblatt.asm kernels/vecadd.s -l          # listing
python -m rosenblatt.asm kernel.s -o kernel.hex -D N=64
```

| Syntax | Meaning |
| --- | --- |
| `; comment`, `# comment`, `// comment` | |
| `label:` | instruction address |
| `.equ NAME, expr` | constant (may be redefined; evaluated in order) |
| `.alias name, r5` | register alias |
| `.rept N[, var] ... .endr` | repeat a block, binding `var` = 0 .. N-1 |
| `.if expr ... [.else ...] .endif` | conditional assembly |
| `r(expr)` | computed register number, for example `mad r(8 + b), r2, r1` |
| `[ra + expr]`, `[ra - expr]`, `[expr]` | memory operands |
| `nop`, `mov rd, ra`, `li rd, any32`, `bra label`, `neg`, `not`, `relu` | pseudo-instructions |

Expressions use Python syntax (`+ - * / % << >> & | ^ ~`, comparisons, `and`,
`or`) over integers and `.equ` names. Kernels are specialized at assembly
time with `-D NAME=value` (from Python: `assemble_file(path, defines={...})`),
much like C++ templates in CUDA. Several kernels can share the instruction
memory: assemble each at its own `base` and combine them with `link()`, then
launch by name (`dev.launch(..., entry="hidden")`).

## Example

```
; c[i] = a[i] + b[i]        args: a, b, c, n
        s2r   r1, %gtid
        s2r   r2, %arg3
        slt   r3, r1, r2          ; i < n ?
        if.nz r3
            shli  r4, r1, 2       ; byte offset
            s2r   r5, %arg0
            add   r5, r5, r4
            ldw   r6, [r5]
            s2r   r5, %arg1
            add   r5, r5, r4
            ldw   r7, [r5]
            add   r6, r6, r7
            s2r   r5, %arg2
            add   r5, r5, r4
            stw   [r5], r6
        endif
        ret
```
