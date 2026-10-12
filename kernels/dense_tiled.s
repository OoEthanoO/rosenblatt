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
; With BCAST, lane b of each warp instead reads image b0 + b's input word, and
; `dp4ab acc_b, x, w, b` takes it from lane b: one shared-memory load and BT
; dot products per weight word, instead of BT loads and BT dot products. The
; tile's rows are padded to RS bytes so the lanes' loads hit different banks.
;
; Layouts (bytes): X is B x K (int8 0..127), bias is NPAD int32, Y is B x NPAD
; (int8 when REQUANTIZE, else int32). W is NPAD x K int8, either row-major
; (INTERLEAVED 0: W[j][k] at j * K + k) or interleaved by words (INTERLEAVED 1:
; W[j][k] at 4 * ((k / 4) * NPAD + j) + k % 4), where the lanes of a warp,
; which work on neighbouring neurons, read neighbouring words.
; K is a multiple of KC and NPAD a multiple of THREADS (the host pads with zeros).
;
; Defines: THREADS, BT (images per block, <= 8), KC (bytes per stage, multiple of 4),
;          LOG_NGROUPS (log2(NPAD / THREADS)), DP4A (0/1), REQUANTIZE (0/1),
;          INTERLEAVED (0/1), BCAST (0/1: needs DP4A and BT == lanes per warp),
;          DBUF (0/1: two tile buffers, one barrier per stage; needs BCAST),
;          with THREADS == BT * KC / 4 (each thread stages one word).
; args:    %arg0 X, %arg1 W, %arg2 bias, %arg3 Y, %arg4 K, %arg5 NPAD, %arg6 shift
; grid:    (B / BT) * (NPAD / THREADS) blocks of THREADS threads

.equ    NPAD, THREADS << LOG_NGROUPS
.equ    RS, KC + 4                    ; padded tile row (BCAST)
.equ    TILEBUF, 256                  ; offset of the second tile buffer (DBUF, >= BT * RS)

.alias  w,    r1          ; weight word (or byte)
.alias  x,    r2          ; input word (or byte)
.alias  pw,   r3          ; address of W[j][k0]
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
        ; pw = &W[j][0]
        s2r   left, %arg4
.if INTERLEAVED
        shli  pw, u, 2
.else
        mul   pw, u, left
.endif
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
.if BCAST
        ; t = where this thread stages its word, u = the tile row of this lane's image
        s2r   x, %tid
        shri  t, x, LOG_KC_WORDS
        muli  t, t, RS
        andi  x, x, KC / 4 - 1
        shli  x, x, 2
        add   t, t, x
        s2r   u, %laneid
        muli  u, u, RS
.endif
.if DBUF
        ; two tile buffers: stage the first tile now; every stage then fetches
        ; the next tile into the other buffer before computing on this one
        ldw   w, [gx]
        sts   [t], w
        addi  gx, gx, KC
        bar
.endif

stage:  ; ---- 1. the block stages a BT x KC tile of X in shared memory
        ldw   w, [gx]
.if DBUF
        xori  x, t, TILEBUF
        sts   [x], w                  ; the next tile, into the other buffer (unused after the last stage)
        addi  gx, gx, KC
.else
.if BCAST
        sts   [t], w                  ; tile row b holds bytes [b*RS, b*RS + KC)
.else
        s2r   x, %tid
        shli  x, x, 2
        sts   [x], w                  ; tile row b holds bytes [b*KC, b*KC + KC)
.endif
        addi  gx, gx, KC
        bar
.endif
        ; ---- 2. multiply: weights from global, inputs from shared (broadcast)
.rept   KC / 4, kw
.if INTERLEAVED
        ldw   w, [pw + kw * 4 * NPAD] ; W[j][k0 + 4kw .. 4kw+3]
.else
        ldw   w, [pw + kw * 4]
.endif
.if BCAST
        lds   x, [u + kw * 4]         ; word kw of image b0 + laneid
.rept   BT, b
        dp4ab r(8 + b), x, w, b       ; image b0 + b's word comes from lane b
.endr
.else
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
.endif
.endr
.if INTERLEAVED
        addi  pw, pw, KC * NPAD
.else
        addi  pw, pw, KC
.endif
.if DBUF
        xori  t, t, TILEBUF           ; swap buffers: the other one holds the next tile
        xori  u, u, TILEBUF
.endif
        bar                           ; nobody overwrites a tile while it is read
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
