; vecadd.s: c[i] = a[i] + b[i] for 32-bit integers. The "hello world" of GPUs.
;
; CUDA equivalent:
;     int i = blockIdx.x * blockDim.x + threadIdx.x;
;     if (i < n) c[i] = a[i] + b[i];
;
; args: %arg0 = a, %arg1 = b, %arg2 = c (byte addresses), %arg3 = n

        s2r   r1, %gtid           ; i
        s2r   r2, %arg3           ; n
        slt   r3, r1, r2          ; r3 = (i < n)
        if.nz r3                  ; threads past the end sit this out
            shli  r4, r1, 2       ; byte offset of element i
            s2r   r5, %arg0
            add   r5, r5, r4
            ldw   r6, [r5]        ; a[i]
            s2r   r5, %arg1
            add   r5, r5, r4
            ldw   r7, [r5]        ; b[i]
            add   r6, r6, r7
            s2r   r5, %arg2
            add   r5, r5, r4
            stw   [r5], r6        ; c[i]
        endif
        ret
