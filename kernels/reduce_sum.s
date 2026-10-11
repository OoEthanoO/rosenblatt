; reduce_sum.s: y[block] = sum of the block's blockDim elements of x.
;
; Classic tree reduction in shared memory: every step halves the number of
; active threads, and a barrier (`bar`, CUDA's __syncthreads) separates the
; steps so nobody reads a partial sum before it is written.
; blockDim must be a power of two.
;
; args: %arg0 = x, %arg1 = y

.alias  tid,    r1
.alias  slot,   r6          ; byte address of my shared-memory slot
.alias  stride, r7

        s2r   tid, %tid
        s2r   r2, %gtid
        s2r   r3, %arg0
        shli  r4, r2, 2
        add   r3, r3, r4
        ldw   r5, [r3]              ; x[gtid]
        shli  slot, tid, 2
        sts   [slot], r5
        bar
        s2r   stride, %ntid
        shri  stride, stride, 1
step:   slt   r8, tid, stride       ; lower half adds in the upper half
        if.nz r8
            shli  r9, stride, 2
            add   r9, r9, slot
            lds   r10, [r9]
            lds   r11, [slot]
            add   r11, r11, r10
            sts   [slot], r11
        endif
        bar                         ; outside the if: every thread must arrive
        shri  stride, stride, 1
        bnz   stride, step          ; uniform branch
        seqi  r8, tid, 0
        if.nz r8                    ; thread 0 writes the block's result
            lds   r10, [r0]
            s2r   r11, %ctaid
            shli  r11, r11, 2
            s2r   r12, %arg1
            add   r11, r11, r12
            stw   [r11], r10
        endif
        ret
