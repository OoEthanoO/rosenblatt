; One thread computes one output of y = W*x + bias.
; x is int8 in 0..127, padded with zeros to a multiple of four bytes for
; aligned DP4A loads. W is signed int8 in one of two layouts:
;   INTERLEAVED 0  row-major, rows padded like x. The lanes of a warp read
;                  words a whole row apart: one memory request each.
;   INTERLEAVED 1  word k of neuron n at W + 4 * (k * NP + n), NP = output
;                  count padded to a multiple of 8. The lanes of a warp read
;                  neighbouring words, which a memory with wide lines
;                  (GMEM_LINE) serves with one request. Needs DP4A.
;
; Defines: DP4A (0/1), REQUANTIZE (0/1), INTERLEAVED (0/1), UNROLL (1/2).
; UNROLL 2 (needs DP4A and a padded width that is a multiple of 8) handles two
; words per loop iteration: fewer loop instructions, and with a register
; scoreboard four loads of one warp in flight at once.
; REQUANTIZE stores clamp((y + 2^(shift-1)) >> shift, 0, 127) as bytes;
; otherwise stores signed int32 logits. Accumulators must not overflow.
; args: x, W, bias, y, padded input width, output count, shift (1..31),
;       4 * NP (INTERLEAVED only)

.alias neuron, r1
.alias px,     r2
.alias pw,     r3
.alias py,     r4
.alias acc,    r5
.alias left,   r6
.alias x,      r7
.alias w,      r8
.alias t,      r9
.alias shift,  r10
.alias wstep,  r11
.alias x1,     r12
.alias w1,     r13
.alias pw1,    r14            ; second weight pointer (UNROLL 2, interleaved)

        s2r   neuron, %gtid
        s2r   t, %arg5
        sltu  t, neuron, t
        if.nz t
            s2r   left, %arg4
.if INTERLEAVED
            shli  pw, neuron, 2
            s2r   wstep, %arg7
.else
            mul   pw, neuron, left
.endif
            s2r   t, %arg1
            add   pw, pw, t
            s2r   px, %arg0
            shli  t, neuron, 2
            s2r   py, %arg2
            add   py, py, t
            ldw   acc, [py]
.if UNROLL == 2 and INTERLEAVED
            add   pw1, pw, wstep
            shli  wstep, wstep, 1
.endif

dot:
.if DP4A and UNROLL == 2
            ldw   x, [px]
            ldw   w, [pw]
            ldw   x1, [px + 4]
.if INTERLEAVED
            ldw   w1, [pw1]
.else
            ldw   w1, [pw + 4]
.endif
            dp4a  acc, x, w
            dp4a  acc, x1, w1
            addi  px, px, 8
.if INTERLEAVED
            add   pw, pw, wstep
            add   pw1, pw1, wstep
.else
            addi  pw, pw, 8
.endif
            subi  left, left, 8
.else
.if DP4A
            ldw   x, [px]
            ldw   w, [pw]
            dp4a  acc, x, w
            addi  px, px, 4
.if INTERLEAVED
            add   pw, pw, wstep
.else
            addi  pw, pw, 4
.endif
            subi  left, left, 4
.else
            ldbu  x, [px]
            ldb   w, [pw]
            mad   acc, x, w
            addi  px, px, 1
            addi  pw, pw, 1
            subi  left, left, 1
.endif
.endif
            bnz   left, dot

            s2r   py, %arg3
.if REQUANTIZE
            s2r   shift, %arg6
            subi  t, shift, 1
            li    x, 1
            shl   x, x, t
            add   acc, acc, x
            sra   acc, acc, shift
            maxi  acc, acc, 0
            mini  acc, acc, 127
            add   py, py, neuron
            stb   [py], acc
.else
            shli  t, neuron, 2
            add   py, py, t
            stw   [py], acc
.endif
        endif
        ret
