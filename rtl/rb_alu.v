// rb_alu.v: one lane's integer ALU (purely combinational).
//
// Every lane of a core has one of these; all lanes see the same `fn` (that is
// what SIMT means: one instruction, many data).
//
// MUL_WIDTH = 32: mul/mad produce the low 32 bits of a 32x32 product.
// MUL_WIDTH = 16: only the low 16 bits of each operand are multiplied (signed),
//                 like NVIDIA's old mul24. One FPGA DSP block per lane instead of 3-4.
`default_nettype none
`include "rb_isa.vh"

module rb_alu #(
    parameter MUL_WIDTH   = 32,
    parameter ENABLE_DP4A = 1
) (
    input  wire [3:0]  fn,
    input  wire [31:0] a,   // rs1
    input  wire [31:0] b,   // rs2 or immediate
    input  wire [31:0] c,   // rd (accumulator for mad/dp4a)
    output reg  [31:0] y
);
    // ---- multiplier (shared by mul and mad)
    wire [31:0] prod;
    generate
        if (MUL_WIDTH == 16) begin : g_mul16
            wire signed [15:0] a16 = a[15:0];
            wire signed [15:0] b16 = b[15:0];
            wire signed [31:0] p16 = a16 * b16;
            assign prod = p16;
        end else begin : g_mul32
            assign prod = a * b;  // only the low 32 bits are kept
        end
    endgenerate

    // ---- packed int8 dot product: c + a.b0*b.b0 + ... + a.b3*b.b3
    wire [31:0] dp4;
    generate
        if (ENABLE_DP4A) begin : g_dp4a
            wire signed [7:0]  a0 = a[7:0],   a1 = a[15:8],  a2 = a[23:16], a3 = a[31:24];
            wire signed [7:0]  b0 = b[7:0],   b1 = b[15:8],  b2 = b[23:16], b3 = b[31:24];
            wire signed [15:0] p0 = a0 * b0,  p1 = a1 * b1,  p2 = a2 * b2,  p3 = a3 * b3;
            assign dp4 = c + {{16{p0[15]}}, p0} + {{16{p1[15]}}, p1}
                           + {{16{p2[15]}}, p2} + {{16{p3[15]}}, p3};
        end else begin : g_no_dp4a
            assign dp4 = 32'd0;
        end
    endgenerate

    wire lt_signed = $signed(a) < $signed(b);

    always @* begin
        case (fn)
            `RB_FN_ADD:  y = a + b;
            `RB_FN_SUB:  y = a - b;
            `RB_FN_MUL:  y = prod;
            `RB_FN_MAD:  y = prod + c;
            `RB_FN_AND:  y = a & b;
            `RB_FN_OR:   y = a | b;
            `RB_FN_XOR:  y = a ^ b;
            `RB_FN_SHL:  y = a << b[4:0];
            `RB_FN_SHR:  y = a >> b[4:0];
            `RB_FN_SRA:  y = $signed(a) >>> b[4:0];
            `RB_FN_SLT:  y = {31'd0, lt_signed};
            `RB_FN_SLTU: y = {31'd0, a < b};
            `RB_FN_SEQ:  y = {31'd0, a == b};
            `RB_FN_MIN:  y = lt_signed ? a : b;
            `RB_FN_MAX:  y = lt_signed ? b : a;
            default:     y = dp4;  // `RB_FN_DP4A
        endcase
    end
endmodule

`default_nettype wire
