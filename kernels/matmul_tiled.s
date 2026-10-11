; matmul_tiled.s: C = A x B for 32-bit integers, tiled through shared memory.
;
; A is M x K, B is K x N, C is M x N, row-major. A block of WARPS x LANES
; threads computes a WARPS x LANES tile of C: thread (warp ty, lane tx) owns
; C[row][col]. K is consumed LANES at a time:
;
;   1. the block copies a WARPS x LANES tile of A and a LANES x LANES tile of
;      B from global into shared memory (each element is fetched once)
;   2. bar
;   3. every thread does LANES multiply-adds reading only shared memory:
;      As[ty][k] is the same for the whole warp (one broadcast read) and
;      Bs[k][tx] hits LANES different banks (no conflicts)
;   4. bar, next tile
;
; Compared to matmul_naive, each global element is reused by many threads, so
; global-memory traffic drops by roughly the tile size.
;
; Requires: M % WARPS == 0, N % LANES == 0, K % LANES == 0, LANES % WARPS == 0.
; Defines:  LANES, LOG_LANES, WARPS, LOG_WARPS (from the device configuration)
; args:     %arg0 = A, %arg1 = B, %arg2 = C, %arg3 = K, %arg4 = N,
;           %arg5 = log2(N / LANES)

.equ    AS_BYTES, WARPS * LANES * 4      ; As occupies shared bytes [0, AS_BYTES)
.alias  ty,  r1
.alias  tx,  r2
.alias  row, r3
.alias  col, r4
.alias  pa,  r5          ; &A[row][k0 + tx]
.alias  pb,  r6          ; &B[k0 + ty][col]
.alias  acc, r7
.alias  kr,  r8          ; K remaining
.alias  t,   r9
.alias  u,   r10
.alias  sa,  r11         ; shared byte address of As[ty][0]
.alias  sb,  r12         ; shared byte address of Bs[0][tx]
.alias  sbt, r13         ; shared byte address of Bs[ty][tx]
.alias  nbw, r14         ; WARPS rows of B, in bytes
.alias  nbt, r15         ; LANES rows of B, in bytes

        s2r   ty, %warpid
        s2r   tx, %laneid
        ; which tile: tile_row = ctaid >> arg5, tile_col = ctaid & (2^arg5 - 1)
        s2r   t, %ctaid
        s2r   u, %arg5
        shr   row, t, u
        li    col, 1
        shl   col, col, u
        subi  col, col, 1
        and   col, t, col
        shli  row, row, LOG_WARPS
        add   row, row, ty
        shli  col, col, LOG_LANES
        add   col, col, tx
        ; pa = A + (row * K + tx) * 4
        s2r   kr, %arg3
        mul   pa, row, kr
        add   pa, pa, tx
        shli  pa, pa, 2
        s2r   t, %arg0
        add   pa, pa, t
        ; pb = B + (ty * N + col) * 4
        s2r   u, %arg4
        mul   pb, ty, u
        add   pb, pb, col
        shli  pb, pb, 2
        s2r   t, %arg1
        add   pb, pb, t
        shli  u, u, 2                    ; N * 4
        shli  nbw, u, LOG_WARPS
        shli  nbt, u, LOG_LANES
        ; shared-memory addresses
        shli  sa, ty, LOG_LANES + 2
        shli  sb, tx, 2
        addi  sb, sb, AS_BYTES
        shli  sbt, ty, LOG_LANES + 2
        add   sbt, sbt, sb
        li    acc, 0

tile:   ; ---- 1. stage the tiles
        ldw   t, [pa]
        shli  u, tx, 2
        add   u, u, sa
        sts   [u], t                     ; As[ty][tx]
        mov   u, pb
.rept   LANES / WARPS, r
        ldw   t, [u]
        sts   [sbt + r * WARPS * LANES * 4], t   ; Bs[ty + r*WARPS][tx]
        add   u, u, nbw
.endr
        bar
        ; ---- 3. multiply out of shared memory
.rept   LANES, k
        lds   t, [sa + k * 4]            ; As[ty][k]   broadcast
        lds   u, [sb + k * LANES * 4]    ; Bs[k][tx]   conflict-free
        mad   acc, t, u
.endr
        bar
        addi  pa, pa, LANES * 4
        add   pb, pb, nbt
        subi  kr, kr, LANES
        bnz   kr, tile

        ; C[row][col] = acc
        s2r   t, %arg4
        mul   t, row, t
        add   t, t, col
        shli  t, t, 2
        s2r   u, %arg2
        add   t, t, u
        stw   [t], acc
        ret
