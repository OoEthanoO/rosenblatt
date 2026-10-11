; One thread computes one output of y = W*x + bias.
; W is row-major signed int8; x is int8 in 0..127. Rows and x are padded
; with zeros to a multiple of four bytes, for aligned DP4A loads.
;
; Defines: DP4A (0/1), REQUANTIZE (0/1).
; REQUANTIZE stores clamp((y + 2^(shift-1)) >> shift, 0, 127) as bytes;
; otherwise stores signed int32 logits. Accumulators must not overflow.
; args: x, W, bias, y, padded input width, output count, shift (1..31)

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

        s2r   neuron, %gtid
        s2r   t, %arg5
        sltu  t, neuron, t
        if.nz t
            s2r   left, %arg4
            mul   pw, neuron, left
            s2r   t, %arg1
            add   pw, pw, t
            s2r   px, %arg0
            shli  t, neuron, 2
            s2r   py, %arg2
            add   py, py, t
            ldw   acc, [py]

dot:
.if DP4A
            ldw   x, [px]
            ldw   w, [pw]
            dp4a  acc, x, w
            addi  px, px, 4
            addi  pw, pw, 4
            subi  left, left, 4
.else
            ldbu  x, [px]
            ldb   w, [pw]
            mad   acc, x, w
            addi  px, px, 1
            addi  pw, pw, 1
            subi  left, left, 1
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
