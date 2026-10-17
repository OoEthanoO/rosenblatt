// rb_alu.v: one lane's integer ALU, split over two pipeline stages.
//
// Every lane of a core has one of these; all lanes see the same `fn` (that is
// what SIMT means: one instruction, many data).
//
//   stage 1 (core stage S2)  operands in; everything except multiply-adds is
//                            computed; multipliers produce partial products
//   stage 2 (core stage S3)  partial products are summed; `y` is the result of
//                            the instruction presented one cycle earlier
//
// The split puts a register between the DSP multipliers and the adders that
// combine their outputs. That halves the longest path in the design (register
// file -> multiplier -> adder tree) and costs no throughput: the core writes
// results back in S3 anyway, and a warp cannot read them before then.
//
// MUL_WIDTH = 32: mul/mad produce the low 32 bits of a 32x32 product, built
//                 from three 16x16 partial products.
// MUL_WIDTH = 16: only the low 16 bits of each operand are multiplied (signed),
//                 like NVIDIA's old mul24. One FPGA DSP block per lane.
//
// The core's matrix unit (mma) borrows the dot-product path (the four 8-bit
// multipliers and their adder tree) in any cycle in which the instruction in
// stage 1 is not a dp4a: dot_sel replaces that path's operands with the
// unit's, and y_dot returns the sum one cycle later. The rest of the ALU keeps
// serving the instruction, so an add, a mul or a compare and one row of a
// matrix product can share a cycle.
`default_nettype none
`include "rb_isa.vh"

module rb_alu #(
    parameter MUL_WIDTH   = 32,
    parameter ENABLE_DP4A = 1
) (
    input  wire        clk,
    // stage 1
    input  wire [3:0]  fn,
    input  wire [31:0] a,       // rs1
    input  wire [31:0] a_dot,   // rs1 for dp4a: this lane's, or one lane's for every lane (dp4ab)
    input  wire [31:0] b,   // rs2 or immediate
    input  wire [31:0] c,   // rd (accumulator for mad/dp4a)
    input  wire        dot_sel, // the matrix unit uses the dot-product path this cycle:
    input  wire [31:0] x_mma,   //   a row of A (the same for every lane),
    input  wire [31:0] w_mma,   //   this lane's column of B,
    input  wire [31:0] c_mma,   //   and the accumulator to add
    // stage 2
    output reg  [31:0] y,
    output wire [31:0] y_dot    // the dot product of the dp4a or mma row that was in stage 1
);
    // ================================================================ stage 1
    // ---- partial products: a * b mod 2^32 = lo + (cross << 16)
    wire [31:0] p_lo;
    wire [15:0] p_cross;
    generate
        if (MUL_WIDTH == 16) begin : g_mul16
            wire signed [15:0] a16 = a[15:0];
            wire signed [15:0] b16 = b[15:0];
            wire signed [31:0] p16 = a16 * b16;
            assign p_lo    = p16;
            assign p_cross = 16'd0;
        end else begin : g_mul32
            wire [31:0] ll = a[15:0] * b[15:0];
            wire [31:0] lh = a[15:0] * b[31:16];
            wire [31:0] hl = a[31:16] * b[15:0];
            assign p_lo    = ll;
            assign p_cross = lh[15:0] + hl[15:0];   // a_hi*b_hi only affects bits >= 32
        end
    endgenerate

    // ---- packed int8 products for dp4a (and mma)
    wire [31:0] da = dot_sel ? x_mma : a_dot;
    wire [31:0] db = dot_sel ? w_mma : b;
    wire [63:0] dprod;
    generate
        if (ENABLE_DP4A) begin : g_dp4a
            wire signed [7:0]  a0 = da[7:0],   a1 = da[15:8];
            wire signed [7:0]  a2 = da[23:16], a3 = da[31:24];
            wire signed [7:0]  b0 = db[7:0],   b1 = db[15:8],  b2 = db[23:16], b3 = db[31:24];
            wire signed [15:0] p0 = a0 * b0,  p1 = a1 * b1,  p2 = a2 * b2,  p3 = a3 * b3;
            assign dprod = {p3, p2, p1, p0};
        end else begin : g_no_dp4a
            assign dprod = 64'd0;
        end
    endgenerate

    // ---- everything that needs no multiplier
    wire       lt_signed = $signed(a) < $signed(b);
    reg [31:0] fast;
    always @* begin
        case (fn)
            `RB_FN_ADD:  fast = a + b;
            `RB_FN_SUB:  fast = a - b;
            `RB_FN_AND:  fast = a & b;
            `RB_FN_OR:   fast = a | b;
            `RB_FN_XOR:  fast = a ^ b;
            `RB_FN_SHL:  fast = a << b[4:0];
            `RB_FN_SHR:  fast = a >> b[4:0];
            `RB_FN_SRA:  fast = $signed(a) >>> b[4:0];
            `RB_FN_SLT:  fast = {31'd0, lt_signed};
            `RB_FN_SLTU: fast = {31'd0, a < b};
            `RB_FN_SEQ:  fast = {31'd0, a == b};
            `RB_FN_MIN:  fast = lt_signed ? a : b;
            `RB_FN_MAX:  fast = lt_signed ? b : a;
            default:     fast = 32'd0;   // mul, mad, dp4a: finished in stage 2
        endcase
    end

    // ---- pipeline register
    reg [3:0]  fn_q;
    reg [31:0] fast_q, c_q, cd_q, lo_q;
    reg [15:0] cross_q;
    reg [63:0] dprod_q;
    always @(posedge clk) begin
        fn_q    <= fn;
        fast_q  <= fast;
        c_q     <= c;                        // mad's addend
        cd_q    <= dot_sel ? c_mma : c;      // the dot product's (the same register without mma)
        lo_q    <= p_lo;
        cross_q <= p_cross;
        dprod_q <= dprod;
    end

    // ================================================================ stage 2
    wire [31:0] prod = lo_q + {cross_q, 16'd0};
    wire [31:0] dsum = cd_q + {{16{dprod_q[15]}}, dprod_q[15:0]}  + {{16{dprod_q[31]}}, dprod_q[31:16]}
                            + {{16{dprod_q[47]}}, dprod_q[47:32]} + {{16{dprod_q[63]}}, dprod_q[63:48]};
    assign y_dot = ENABLE_DP4A ? dsum : 32'd0;
    always @* begin
        case (fn_q)
            `RB_FN_MUL:  y = prod;
            `RB_FN_MAD:  y = prod + c_q;
            `RB_FN_DP4A: y = ENABLE_DP4A ? dsum : 32'd0;
            default:     y = fast_q;
        endcase
    end
endmodule

`default_nettype wire
