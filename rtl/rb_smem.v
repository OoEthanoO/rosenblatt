// rb_smem.v: one core's shared memory (CUDA __shared__), W banks.
//
// Word i lives in bank i % W. Every cycle each bank can serve one row, so a
// warp whose lanes touch different banks completes in one access
// ("wavefront"). Loads from the same word share the read (broadcast). Lanes
// that want *different* rows of the *same* bank conflict: this wavefront serves
// the lowest such lane's row, and the core replays the instruction for the
// lanes left over.
//
// Stores write one lane per bank per wavefront (the lowest waiting one), so
// several lanes storing into the same word take turns. Serving them in lane
// order makes the highest lane's data land last, which is exactly the
// architectural rule ("highest lane wins"), without a byte-merging network.
//
// Timing: addresses and store data arrive in pipeline stage S2; stores are
// written at the end of S2; load data (already byte-extracted per lane) is on
// `rdata` during S3, when the core writes it to the register files.
//
// The routing is one-hot AND-OR logic: exactly one lane leads each bank, so
// "take the leader's row / data" never needs a priority mux. (A first version
// built from priority muxes and variable part-selects was 3x larger.)
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
    output wire [W-1:0]    covered,  // lanes served by this wavefront
    output wire [W*32-1:0] rdata     // S3: load result per lane
);
    localparam LB = $clog2(W);
    localparam RW = SMEM_AW - LB;

    // ---- per-lane fields of the byte address: | row | bank | byte |
    wire [W*LB-1:0] bank;
    wire [W*RW-1:0] row;
    genvar gi, gj, gb;
    generate
        for (gi = 0; gi < W; gi = gi + 1) begin : g_fields
            assign bank[gi*LB +: LB] = addr[gi*32+2 +: LB];
            assign row[gi*RW +: RW]  = addr[gi*32+2+LB +: RW];
        end
    endgenerate

    // ---- leaders: the lowest waiting lane of each bank
    wire [W-1:0] first;
    generate
        for (gi = 0; gi < W; gi = gi + 1) begin : g_first
            wire [W-1:0] lower_same;   // a lower lane waits on the same bank
            wire [W-1:0] blocked;      // ...and leads it with a different row
            for (gj = 0; gj < W; gj = gj + 1) begin : g_pair
                if (gj < gi) begin : g_lt
                    wire same = bank[gj*LB +: LB] == bank[gi*LB +: LB];
                    assign lower_same[gj] = lanes[gj] && same;
                    assign blocked[gj]    = first[gj] && same && row[gj*RW +: RW] != row[gi*RW +: RW];
                end else begin : g_ge
                    assign lower_same[gj] = 1'b0;
                    assign blocked[gj]    = 1'b0;
                end
            end
            assign first[gi]   = lanes[gi] && !(|lower_same);
            // loads ride along with their bank's leader if they want the same row
            assign covered[gi] = store ? first[gi] : (lanes[gi] && !(|blocked));
        end
    endgenerate

    // ---- what each bank does this wavefront: AND-OR over its (one) leader
    wire [W*RW-1:0] b_row;
    wire [W-1:0]    b_found;
    wire [W*32-1:0] b_wdata;
    wire [W*4-1:0]  b_wstrb;
    generate
        for (gb = 0; gb < W; gb = gb + 1) begin : g_lead
            wire [W-1:0] lead;   // one-hot (or zero): which lane leads bank gb
            for (gi = 0; gi < W; gi = gi + 1) begin : g_l
                assign lead[gi] = first[gi] && bank[gi*LB +: LB] == gb;
            end
            reg [RW-1:0] r;
            reg [31:0]   d;
            reg [3:0]    st;
            always @* begin : p_andor
                integer i;
                r  = {RW{1'b0}};
                d  = 32'd0;
                st = 4'd0;
                for (i = 0; i < W; i = i + 1) begin
                    r  = r  | ({RW{lead[i]}} & row[i*RW +: RW]);
                    d  = d  | ({32{lead[i]}} & ((size == 2'b00) ? wdata[i*32 +: 32]
                                                                : {4{wdata[i*32 +: 8]}}));
                    st = st | ({4{lead[i]}} & ((size == 2'b00) ? 4'hF : (4'b0001 << addr[i*32 +: 2])));
                end
            end
            assign b_row[gb*RW +: RW]   = r;
            assign b_found[gb]          = |lead;
            assign b_wdata[gb*32 +: 32] = d;
            assign b_wstrb[gb*4 +: 4]   = st;
        end
    endgenerate

    // ---- the banks
    wire [W*32-1:0] bank_q;
    generate
        for (gb = 0; gb < W; gb = gb + 1) begin : g_bank
            reg  [31:0]   mem [0:(1<<RW)-1];
            reg  [31:0]   q;
            wire [RW-1:0] ra = b_row[gb*RW +: RW];
            wire          we = access && store && b_found[gb];
            integer k;
`ifndef SYNTHESIS
            initial for (k = 0; k < (1 << RW); k = k + 1) mem[k] = 32'd0;
`endif
            always @(posedge clk) begin
                if (we) begin
                    if (b_wstrb[gb*4+0]) mem[ra][7:0]   <= b_wdata[gb*32+0  +: 8];
                    if (b_wstrb[gb*4+1]) mem[ra][15:8]  <= b_wdata[gb*32+8  +: 8];
                    if (b_wstrb[gb*4+2]) mem[ra][23:16] <= b_wdata[gb*32+16 +: 8];
                    if (b_wstrb[gb*4+3]) mem[ra][31:24] <= b_wdata[gb*32+24 +: 8];
                end
                else q <= mem[ra];   // stores need no read data
            end
            assign bank_q[gb*32 +: 32] = q;
        end
    endgenerate

    // ---- S3: route each lane's bank word back (one-hot) and extract the byte
    reg [W*W-1:0] bank_oh;   // bank_oh[i*W + b]: lane i read bank b
    reg [W*2-1:0] off_q;
    reg [1:0]     size_q;
    always @(posedge clk) begin : p_route
        integer i, b;
        for (i = 0; i < W; i = i + 1) begin
            for (b = 0; b < W; b = b + 1) bank_oh[i*W + b] <= bank[i*LB +: LB] == b;
            off_q[i*2 +: 2] <= addr[i*32 +: 2];
        end
        size_q <= size;
    end

    generate
        for (gi = 0; gi < W; gi = gi + 1) begin : g_lane
            reg [31:0] w;
            always @* begin : p_pick
                integer b;
                w = 32'd0;
                for (b = 0; b < W; b = b + 1)
                    w = w | ({32{bank_oh[gi*W + b]}} & bank_q[b*32 +: 32]);
            end
            reg  [7:0]  by;
            always @* begin
                case (off_q[gi*2 +: 2])
                    2'd0: by = w[7:0];
                    2'd1: by = w[15:8];
                    2'd2: by = w[23:16];
                    default: by = w[31:24];
                endcase
            end
            assign rdata[gi*32 +: 32] = (size_q == 2'b00) ? w :
                                        (size_q == 2'b01) ? {{24{by[7]}}, by} : {24'd0, by};
        end
    endgenerate
endmodule

`default_nettype wire
