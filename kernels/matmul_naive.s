; matmul_naive.s: C = A x B for 32-bit integers, one thread per element of C.
;
; A is M x K, B is K x N, C is M x N, all row-major.
; Thread g = blockIdx.x * blockDim.x + threadIdx.x computes C[g / N][g % N]
; (N must be a power of two: the kernel uses shifts instead of division).
;
; Every multiply-add needs two loads from global memory, so this kernel
; is limited by memory bandwidth, not by the ALUs.
;
; args: %arg0 = A, %arg1 = B, %arg2 = C, %arg3 = M*N, %arg4 = K, %arg5 = log2(N)

.alias  row,  r1
.alias  col,  r2
.alias  pa,   r3          ; &A[row][k]
.alias  pb,   r4          ; &B[k][col]
.alias  acc,  r5
.alias  k,    r6
.alias  a,    r7
.alias  b,    r8
.alias  n4,   r9          ; row stride of B in bytes (4 * N)
.alias  t,    r10
.alias  kk,   r11

        s2r   r12, %gtid
        s2r   t, %arg3
        slt   t, r12, t
        if.nz t
            s2r   t, %arg5            ; log2(N)
            shr   row, r12, t         ; row = g >> log2N
            li    n4, 1
            shl   n4, n4, t           ; N
            subi  col, n4, 1
            and   col, r12, col       ; col = g & (N-1)
            shli  n4, n4, 2           ; 4*N
            s2r   kk, %arg4           ; K
            ; pa = A + row * K * 4
            mul   pa, row, kk
            shli  pa, pa, 2
            s2r   t, %arg0
            add   pa, pa, t
            ; pb = B + col * 4
            shli  pb, col, 2
            s2r   t, %arg1
            add   pb, pb, t
            li    acc, 0
            mov   k, kk
inner:      ldw   a, [pa]
            ldw   b, [pb]
            mad   acc, a, b           ; acc += a * b
            addi  pa, pa, 4
            add   pb, pb, n4
            subi  k, k, 1
            bnz   k, inner            ; uniform: every thread has the same K
            ; C[row][col]
            shli  t, r12, 2
            s2r   a, %arg2
            add   t, t, a
            stw   [t], acc
        endif
        ret
