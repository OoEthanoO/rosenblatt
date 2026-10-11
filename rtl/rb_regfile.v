// rb_regfile.v: one lane's register file: NUM_WARPS x 16 registers of 32 bits.
//
// A core runs several warps, so each lane keeps a private copy of r0..r15 for
// every warp it hosts; the address is {warp, reg}.
//
// An instruction reads up to three registers at once (rs1, rs2 and rd: rd is
// the accumulator of mad/dp4a and the data register of stores). FPGA RAMs have
// one read port, so the file is simply stored three times; every write goes to
// all three copies. Reads are synchronous (data appears the next cycle), which
// lets synthesis use block RAM or LUT RAM. r0 always reads as zero.
`default_nettype none

module rb_regfile #(
    parameter AW = 6  // log2(NUM_WARPS) + 4
) (
    input  wire          clk,
    input  wire [AW-1:0] ra,
    input  wire [AW-1:0] rb,
    input  wire [AW-1:0] rc,
    output wire [31:0]   qa,
    output wire [31:0]   qb,
    output wire [31:0]   qc,
    input  wire          we,
    input  wire [AW-1:0] wa,
    input  wire [31:0]   wd
);
    reg [31:0] copy_a [0:(1<<AW)-1];
    reg [31:0] copy_b [0:(1<<AW)-1];
    reg [31:0] copy_c [0:(1<<AW)-1];
    reg [31:0] qa_r, qb_r, qc_r;
    reg        za, zb, zc;

`ifndef SYNTHESIS  // simulation only: avoid X before a register is first written
    integer i;
    initial begin
        for (i = 0; i < (1 << AW); i = i + 1) begin
            copy_a[i] = 32'd0;
            copy_b[i] = 32'd0;
            copy_c[i] = 32'd0;
        end
    end
`endif

    always @(posedge clk) begin
        if (we) begin
            copy_a[wa] <= wd;
            copy_b[wa] <= wd;
            copy_c[wa] <= wd;
        end
        qa_r <= copy_a[ra];
        qb_r <= copy_b[rb];
        qc_r <= copy_c[rc];
        za   <= ra[3:0] == 4'd0;
        zb   <= rb[3:0] == 4'd0;
        zc   <= rc[3:0] == 4'd0;
    end

    assign qa = za ? 32'd0 : qa_r;
    assign qb = zb ? 32'd0 : qb_r;
    assign qc = zc ? 32'd0 : qc_r;
endmodule

`default_nettype wire
