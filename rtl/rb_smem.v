// rb_smem.v: one core's shared memory (CUDA __shared__), W banks.
//
// Word i lives in bank i % W. Every cycle each bank can serve one row, so a
// warp whose lanes touch different banks, or the same word, completes in one
// access ("wavefront"). Lanes that want *different* rows of the *same* bank
// conflict: this wavefront serves the lowest such lane's row, and the core
// replays the instruction for the lanes left over.
//
// Timing: addresses and store data arrive in pipeline stage S2; stores are
// written at the end of S2; load data (already byte-extracted per lane) is on
// `rdata` during S3, when the core writes it to the register files.
`default_nettype none

module rb_smem #(
    parameter W       = 8,    // lanes = banks (power of two, >= 2)
    parameter SMEM_AW = 10    // log2(words), > log2(W)
) (
    input  wire            clk,
    input  wire            access,   // perform one wavefront this cycle
    input  wire            store,
    input  wire [1:0]      size,     // 00 word, 01 signed byte, 10 unsigned byte
    input  wire [W-1:0]    lanes,    // lanes still waiting for this instruction
    input  wire [W*32-1:0] addr,     // per-lane byte addresses
    input  wire [W*32-1:0] wdata,    // per-lane store data
    output reg  [W-1:0]    covered,  // lanes served by this wavefront
    output wire [W*32-1:0] rdata     // S3: load result per lane
);
    localparam LB = $clog2(W);
    localparam RW = SMEM_AW - LB;

    // ---- bank arbitration: each bank serves the row of its lowest waiting lane
    reg [W-1:0]    b_found;
    reg [W*RW-1:0] b_row;
    always @* begin : p_select
        integer i, b;
        b_found = {W{1'b0}};
        b_row   = {W*RW{1'b0}};
        for (b = 0; b < W; b = b + 1)
            for (i = W - 1; i >= 0; i = i - 1)
                if (lanes[i] && addr[i*32+2 +: LB] == b) begin
                    b_found[b]        = 1'b1;
                    b_row[b*RW +: RW] = addr[i*32+2+LB +: RW];
                end
    end

    always @* begin : p_cover
        integer i;
        for (i = 0; i < W; i = i + 1)
            covered[i] = lanes[i] &&
                (addr[i*32+2+LB +: RW] == b_row[addr[i*32+2 +: LB]*RW +: RW]);
    end

    // ---- store data per bank (ascending lane order: the highest lane wins)
    reg [W*32-1:0] b_wdata;
    reg [W*4-1:0]  b_wstrb;
    always @* begin : p_merge
        integer i;
        b_wdata = {W*32{1'b0}};
        b_wstrb = {W*4{1'b0}};
        for (i = 0; i < W; i = i + 1) begin
            if (covered[i]) begin
                if (size == 2'b00) begin
                    b_wdata[addr[i*32+2 +: LB]*32 +: 32] = wdata[i*32 +: 32];
                    b_wstrb[addr[i*32+2 +: LB]*4 +: 4]   = 4'hF;
                end else begin
                    b_wdata[addr[i*32+2 +: LB]*32 + addr[i*32 +: 2]*8 +: 8] = wdata[i*32 +: 8];
                    b_wstrb[addr[i*32+2 +: LB]*4 + addr[i*32 +: 2]]         = 1'b1;
                end
            end
        end
    end

    // ---- the banks
    wire [W*32-1:0] bank_q;
    genvar gb;
    generate
        for (gb = 0; gb < W; gb = gb + 1) begin : g_bank
            reg  [31:0]   mem [0:(1<<RW)-1];
            reg  [31:0]   q;
            wire [RW-1:0] row = b_row[gb*RW +: RW];
            wire          we  = access && store && b_found[gb];
            integer k;
            initial for (k = 0; k < (1 << RW); k = k + 1) mem[k] = 32'd0;
            always @(posedge clk) begin
                if (we) begin
                    if (b_wstrb[gb*4+0]) mem[row][7:0]   <= b_wdata[gb*32+0  +: 8];
                    if (b_wstrb[gb*4+1]) mem[row][15:8]  <= b_wdata[gb*32+8  +: 8];
                    if (b_wstrb[gb*4+2]) mem[row][23:16] <= b_wdata[gb*32+16 +: 8];
                    if (b_wstrb[gb*4+3]) mem[row][31:24] <= b_wdata[gb*32+24 +: 8];
                end
                q <= mem[row];
            end
            assign bank_q[gb*32 +: 32] = q;
        end
    endgenerate

    // ---- S3: route each lane's bank word back and extract the byte
    reg [W*LB-1:0] bank_of;   // which bank each lane read
    reg [W*2-1:0]  off_of;    // byte offset inside the word
    reg [1:0]      size_q;
    always @(posedge clk) begin : p_route
        integer i;
        for (i = 0; i < W; i = i + 1) begin
            bank_of[i*LB +: LB] <= addr[i*32+2 +: LB];
            off_of[i*2 +: 2]    <= addr[i*32 +: 2];
        end
        size_q <= size;
    end

    genvar gl;
    generate
        for (gl = 0; gl < W; gl = gl + 1) begin : g_lane
            wire [31:0] w = bank_q[bank_of[gl*LB +: LB]*32 +: 32];
            reg  [7:0]  b;
            always @* begin
                case (off_of[gl*2 +: 2])
                    2'd0: b = w[7:0];
                    2'd1: b = w[15:8];
                    2'd2: b = w[23:16];
                    default: b = w[31:24];
                endcase
            end
            assign rdata[gl*32 +: 32] = (size_q == 2'b00) ? w :
                                        (size_q == 2'b01) ? {{24{b[7]}}, b} : {24'd0, b};
        end
    endgenerate
endmodule

`default_nettype wire
