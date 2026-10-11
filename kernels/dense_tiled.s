; dense_tiled.s: a batch of dense int8 layers, Y = act(X * W^T + bias), tiled
; through shared memory. This is the "add a shared memory" kernel of the
; profiling study; compare with dense_int8.s, which runs one image at a time.
;
; A block of THREADS threads owns THREADS output neurons for BT images:
;   thread t computes neuron j = n0 + t for images b0 .. b0+BT-1, keeping one
;   accumulator per image in registers r8.. (register blocking: each weight
;   word fetched from global memory is reused BT times).
; The input is consumed KC bytes at a time. Each step, the whole block copies
; a BT x KC tile of X into shared memory (one word per thread), waits at a
; barrier, and then every thread reads the inputs from shared memory. All
; lanes of a warp read the same input word, so each read is a single
; broadcast wavefront.
;
; Layouts (bytes): X is B x K (int8 0..127), W is NPAD x K row-major int8,
; bias is NPAD int32, Y is B x NPAD (int8 when REQUANTIZE, else int32).
; K is a multiple of KC and NPAD a multiple of THREADS (the host pads with zeros).
;
; Defines: THREADS, BT (images per block, <= 8), KC (bytes per stage, multiple of 4),
;          LOG_NGROUPS (log2(NPAD / THREADS)), DP4A (0/1), REQUANTIZE (0/1)
;          with THREADS == BT * KC / 4 (each thread stages one word).
; args:    %arg0 X, %arg1 W, %arg2 bias, %arg3 Y, %arg4 K, %arg5 NPAD, %arg6 shift
; grid:    (B / BT) * (NPAD / THREADS) blocks of THREADS threads

.alias  w,    r1          ; weight word (or byte)
.alias  x,    r2          ; input word (or byte)
.alias  pw,   r3          ; &W[j][k0]
.alias  gx,   r4          ; this thread's word of the next X tile in global memory
.alias  left, r5          ; stages remaining
.alias  t,    r6
.alias  u,    r7
                          ; r8 .. r8+BT-1: accumulators for images b0 .. b0+BT-1

        ; ---- which tile: neuron group (low bits of ctaid) and image group
        s2r   t, %ctaid
        andi  u, t, (1 << LOG_NGROUPS) - 1
        muli  u, u, THREADS
        s2r   x, %tid
        add   u, u, x                 ; u = j, this thread's neuron
        shri  t, t, LOG_NGROUPS
        muli  t, t, BT                ; t = b0, first image of the tile
        ; pw = W + j * K
        s2r   left, %arg4
        mul   pw, u, left
        s2r   w, %arg1
        add   pw, pw, w
        ; every accumulator starts at bias[j]
        shli  w, u, 2
        s2r   x, %arg2
        add   w, w, x
        ldw   w, [w]
.rept   BT, b
        mov   r(8 + b), w
.endr
        ; gx = X + (b0 + tid / (KC/4)) * K + 4 * (tid % (KC/4))
        s2r   x, %tid
        shri  w, x, LOG_KC_WORDS
        add   w, w, t
        mul   gx, w, left
        andi  w, x, KC / 4 - 1
        shli  w, w, 2
        add   gx, gx, w
        s2r   w, %arg0
        add   gx, gx, w
        shri  left, left, LOG_KC      ; stages = K / KC

stage:  ; ---- 1. the block stages a BT x KC tile of X in shared memory
        ldw   w, [gx]
        s2r   x, %tid
        shli  x, x, 2
        sts   [x], w                  ; tile row b holds bytes [b*KC, b*KC + KC)
        addi  gx, gx, KC
        bar
        ; ---- 2. multiply: weights from global, inputs from shared (broadcast)
.rept   KC / 4, kw
        ldw   w, [pw + kw * 4]        ; W[j][k0 + 4kw .. 4kw+3]
.if DP4A
.rept   BT, b
        lds   x, [r0 + b * KC + kw * 4]
        dp4a  r(8 + b), x, w
.endr
.else
.rept   4, k
        shli  u, w, 24 - 8 * k        ; sign-extend weight byte k
        srai  u, u, 24
.rept   BT, b
        ldsbu x, [r0 + b * KC + kw * 4 + k]
        mad   r(8 + b), x, u
.endr
.endr
.endif
.endr
        addi  pw, pw, KC
        bar                           ; nobody overwrites the tile while it is read
        subi  left, left, 1
        bnz   left, stage

        ; ---- 3. epilogue: Y[b0 + b][j]
        s2r   t, %ctaid
        andi  u, t, (1 << LOG_NGROUPS) - 1
        muli  u, u, THREADS
        s2r   x, %tid
        add   u, u, x                 ; j
        shri  t, t, LOG_NGROUPS
        muli  t, t, BT                ; b0
        s2r   w, %arg5                ; NPAD
        mul   t, t, w
        add   t, t, u                 ; element index of Y[b0][j]
        s2r   x, %arg3
.if REQUANTIZE
        add   t, t, x                 ; byte address
        s2r   pw, %arg6               ; shift
        subi  gx, pw, 1
        li    left, 1
        shl   left, left, gx          ; rounding constant 2^(shift-1)
.rept   BT, b
        add   u, r(8 + b), left
        sra   u, u, pw
        maxi  u, u, 0
        mini  u, u, 127
        stb   [t], u
        add   t, t, w                 ; next image row
.endr
.else
        shli  t, t, 2
        add   t, t, x                 ; word address
        shli  w, w, 2
.rept   BT, b
        stw   [t], r(8 + b)
        add   t, t, w
.endr
.endif
        ret
