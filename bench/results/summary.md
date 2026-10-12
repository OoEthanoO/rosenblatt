# Profiling results

Cycle-accurate RTL simulation (Icarus Verilog) of MNIST inference, checked logit-for-logit against the NumPy reference. Default GPU: 2 cores x 8 lanes x 4 warps, global memory latency 8 cycles, one 32-bit request per cycle.

## Inference strategies

| Experiment | cycles / image | warp instructions / image | global requests / image | IPC per core | memory port busy | issue slots lost to memory waits | replays / image | correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| per_image_scalar | 70,324 | 45,071 | 57,316 | 0.33 | 82% | 24% | 58,736 | 4/4 |
| per_image_dp4a | 17,796 | 11,471 | 14,404 | 0.33 | 81% | 24% | 14,736 | 4/4 |
| tiled_scalar | 8,851 | 15,532 | 2,100 | 0.88 | 24% | 5% | 870 | 32/32 |
| tiled_dp4a | 3,139 | 4,108 | 2,100 | 0.65 | 67% | 14% | 870 | 32/32 |

## Memory latency (per_image)

Default latency is 8 cycles.

| Experiment | cycles / image | warp instructions / image | global requests / image | IPC per core | memory port busy | issue slots lost to memory waits | replays / image | correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| latency1_per_image | 16,294 | 11,471 | 14,404 | 0.36 | 88% | 17% | 14,572 | 4/4 |
| latency4_per_image | 17,993 | 11,471 | 14,404 | 0.33 | 80% | 22% | 14,139 | 4/4 |
| per_image_dp4a | 17,796 | 11,471 | 14,404 | 0.33 | 81% | 24% | 14,736 | 4/4 |
| latency16_per_image | 17,314 | 11,471 | 14,404 | 0.34 | 83% | 51% | 4,871 | 4/4 |
| latency32_per_image | 22,667 | 11,471 | 14,404 | 0.26 | 64% | 70% | 1,682 | 4/4 |

## Memory bandwidth (per_image)

intervalN: the memory accepts one request every N cycles.

| Experiment | cycles / image | warp instructions / image | global requests / image | IPC per core | memory port busy | issue slots lost to memory waits | replays / image | correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| per_image_dp4a | 17,796 | 11,471 | 14,404 | 0.33 | 81% | 24% | 14,736 | 4/4 |
| interval2_per_image | 29,486 | 11,471 | 14,404 | 0.20 | 98% | 17% | 36,314 | 4/4 |
| interval4_per_image | 58,104 | 11,471 | 14,404 | 0.10 | 99% | 9% | 93,045 | 4/4 |

## Cores (per_image)

| Experiment | cycles / image | warp instructions / image | global requests / image | IPC per core | memory port busy | issue slots lost to memory waits | replays / image | correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cores1_per_image | 28,620 | 11,471 | 14,404 | 0.40 | 50% | 26% | 9,463 | 4/4 |
| per_image_dp4a | 17,796 | 11,471 | 14,404 | 0.33 | 81% | 24% | 14,736 | 4/4 |
| cores4_per_image | 17,796 | 11,471 | 14,404 | 0.33 | 81% | 24% | 14,736 | 4/4 |

## Load/store unit design (per_image)

lsu1: one global memory instruction in flight per core (the first design); park: warps sleep while the LSU is busy instead of retrying.

| Experiment | cycles / image | warp instructions / image | global requests / image | IPC per core | memory port busy | issue slots lost to memory waits | replays / image | correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lsu1_per_image | 30,328 | 11,471 | 14,404 | 0.19 | 47% | 6% | 44,129 | 4/4 |
| lsu1_park_per_image | 34,267 | 11,471 | 14,404 | 0.17 | 42% | 71% | 7,906 | 4/4 |
| per_image_dp4a | 17,796 | 11,471 | 14,404 | 0.33 | 81% | 24% | 14,736 | 4/4 |
| park_per_image | 19,371 | 11,471 | 14,404 | 0.30 | 74% | 57% | 4,734 | 4/4 |

## Memory latency (tiled)

Default latency is 8 cycles.

| Experiment | cycles / image | warp instructions / image | global requests / image | IPC per core | memory port busy | issue slots lost to memory waits | replays / image | correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| latency1_tiled | 2,895 | 4,108 | 2,100 | 0.71 | 73% | 8% | 819 | 32/32 |
| latency4_tiled | 2,985 | 4,108 | 2,100 | 0.69 | 70% | 10% | 819 | 32/32 |
| tiled_dp4a | 3,139 | 4,108 | 2,100 | 0.65 | 67% | 14% | 870 | 32/32 |
| latency16_tiled | 3,344 | 4,108 | 2,100 | 0.61 | 63% | 20% | 845 | 32/32 |
| latency32_tiled | 3,838 | 4,108 | 2,100 | 0.54 | 55% | 31% | 819 | 32/32 |

## Memory bandwidth (tiled)

intervalN: the memory accepts one request every N cycles.

| Experiment | cycles / image | warp instructions / image | global requests / image | IPC per core | memory port busy | issue slots lost to memory waits | replays / image | correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiled_dp4a | 3,139 | 4,108 | 2,100 | 0.65 | 67% | 14% | 870 | 32/32 |
| interval2_tiled | 4,851 | 4,108 | 2,100 | 0.42 | 87% | 16% | 3,511 | 32/32 |
| interval4_tiled | 8,993 | 4,108 | 2,100 | 0.23 | 93% | 17% | 10,163 | 32/32 |

## Cores (tiled)

| Experiment | cycles / image | warp instructions / image | global requests / image | IPC per core | memory port busy | issue slots lost to memory waits | replays / image | correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cores1_tiled | 5,601 | 4,108 | 2,100 | 0.73 | 37% | 13% | 491 | 32/32 |
| tiled_dp4a | 3,139 | 4,108 | 2,100 | 0.65 | 67% | 14% | 870 | 32/32 |
| cores4_tiled | 2,426 | 4,108 | 2,100 | 0.42 | 87% | 16% | 3,510 | 32/32 |

## Load/store unit design (tiled)

lsu1: one global memory instruction in flight per core (the first design); park: warps sleep while the LSU is busy instead of retrying.

| Experiment | cycles / image | warp instructions / image | global requests / image | IPC per core | memory port busy | issue slots lost to memory waits | replays / image | correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lsu1_tiled | 4,083 | 4,108 | 2,100 | 0.50 | 51% | 11% | 2,640 | 32/32 |
| lsu1_park_tiled | 4,199 | 4,108 | 2,100 | 0.49 | 50% | 39% | 481 | 32/32 |
| tiled_dp4a | 3,139 | 4,108 | 2,100 | 0.65 | 67% | 14% | 870 | 32/32 |
| park_tiled | 3,168 | 4,108 | 2,100 | 0.65 | 66% | 26% | 173 | 32/32 |

## Warps per core (per_image)

| Experiment | cycles / image | warp instructions / image | global requests / image | IPC per core | memory port busy | issue slots lost to memory waits | replays / image | correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| warps2_per_image | 23,897 | 11,457 | 14,404 | 0.24 | 60% | 68% | 1,630 | 4/4 |
| per_image_dp4a | 17,796 | 11,471 | 14,404 | 0.33 | 81% | 24% | 14,736 | 4/4 |

## Memory line size and weight layout (per_image)

lineN: one request moves an N-byte line (default 4); rows: weights row-major (default); interleaved: neighbouring neurons' weight words side by side.

| Experiment | cycles / image | warp instructions / image | global requests / image | IPC per core | memory port busy | issue slots lost to memory waits | replays / image | correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| per_image_dp4a | 17,796 | 11,471 | 14,404 | 0.33 | 81% | 24% | 14,736 | 4/4 |
| interleaved_per_image | 17,803 | 11,481 | 14,404 | 0.33 | 81% | 24% | 14,736 | 4/4 |
| line16_rows_per_image | 17,736 | 11,471 | 14,334 | 0.33 | 81% | 24% | 14,654 | 4/4 |
| line16_interleaved_per_image | 10,319 | 11,481 | 4,814 | 0.58 | 47% | 38% | 810 | 4/4 |
| line32_rows_per_image | 17,726 | 11,471 | 14,324 | 0.33 | 81% | 24% | 14,638 | 4/4 |
| line32_interleaved_per_image | 9,349 | 11,481 | 3,220 | 0.64 | 34% | 35% | 23 | 4/4 |

## Memory line size and weight layout (tiled)

lineN: one request moves an N-byte line (default 4); rows: weights row-major (default); interleaved: neighbouring neurons' weight words side by side.

| Experiment | cycles / image | warp instructions / image | global requests / image | IPC per core | memory port busy | issue slots lost to memory waits | replays / image | correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiled_dp4a | 3,139 | 4,108 | 2,100 | 0.65 | 67% | 14% | 870 | 32/32 |
| interleaved_tiled | 3,139 | 4,108 | 2,100 | 0.65 | 67% | 14% | 870 | 32/32 |
| line16_rows_tiled | 2,839 | 4,108 | 1,753 | 0.72 | 62% | 12% | 518 | 32/32 |
| line16_interleaved_tiled | 2,546 | 4,108 | 529 | 0.81 | 21% | 11% | 174 | 32/32 |
| line32_rows_tiled | 2,835 | 4,108 | 1,748 | 0.72 | 62% | 12% | 510 | 32/32 |
| line32_interleaved_tiled | 2,510 | 4,108 | 320 | 0.82 | 13% | 11% | 102 | 32/32 |
