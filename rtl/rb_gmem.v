// rb_gmem.v: global memory, shared by every core and the host.
//
// One 32-bit word per request, at most one request every GMEM_INTERVAL cycles
// for the whole device: this is the bandwidth bottleneck the profiling
// chapter is about. Reads return LATENCY cycles after they are accepted,
// in order; a round-robin arbiter picks among the requesting ports.
// Writes take effect when accepted (byte enables) and send no response.
//
// On an FPGA the array maps to block RAM; the extra latency stages model the
// DRAM a real GPU would talk to.
`default_nettype none

module rb_gmem #(
    parameter NP       = 3,   // ports: cores + host
    parameter AW       = 16,  // log2(words)
    parameter LATENCY  = 8,   // >= 1
    parameter INTERVAL = 1    // >= 1
) (
    input  wire             clk,
    input  wire             rst,
    input  wire [NP-1:0]    req_valid,
    output wire [NP-1:0]    req_ready,
    input  wire [NP-1:0]    req_we,
    input  wire [NP*AW-1:0] req_addr,
    input  wire [NP*32-1:0] req_wdata,
    input  wire [NP*4-1:0]  req_wstrb,
    output wire [NP-1:0]    resp_valid,
    output wire [31:0]      resp_rdata,
    output wire [NP-1:0]    accepted   // which port was served this cycle
);
    localparam PB = (NP > 1) ? $clog2(NP) : 1;

    // ---- arbitration
    reg  [PB-1:0] rr_last;
    reg  [PB-1:0] sel;
    reg           sel_ok;
    always @* begin : p_arb
        integer k, cand;
        sel    = rr_last;
        sel_ok = 1'b0;
        for (k = NP; k >= 1; k = k - 1) begin
            cand = rr_last + k;
            if (cand >= NP) cand = cand - NP;
            if (req_valid[cand]) begin
                sel    = cand[PB-1:0];
                sel_ok = 1'b1;
            end
        end
    end

    reg  [31:0] cool;  // cycles until the next request may be accepted
    wire        open = (cool == 32'd0);
    wire        acc  = open && sel_ok;
    wire [NP-1:0] grant = acc ? ({{(NP-1){1'b0}}, 1'b1} << sel) : {NP{1'b0}};
    assign req_ready = grant;
    assign accepted  = grant;

    wire          we    = req_we[sel];
    wire [AW-1:0] addr  = req_addr[sel*AW +: AW];
    wire [31:0]   wdata = req_wdata[sel*32 +: 32];
    wire [3:0]    wstrb = req_wstrb[sel*4 +: 4];

    always @(posedge clk) begin
        if (rst) begin
            rr_last <= {PB{1'b0}};
            cool    <= 32'd0;
        end else begin
            if (acc) rr_last <= sel;
            if (acc)        cool <= INTERVAL - 1;
            else if (!open) cool <= cool - 1;
        end
    end

    // ---- storage
    reg [31:0] mem [0:(1<<AW)-1];
    reg [31:0] q;
    integer ii;
`ifndef SYNTHESIS  // simulation only: FPGA block RAM powers up as zero
    initial for (ii = 0; ii < (1 << AW); ii = ii + 1) mem[ii] = 32'd0;
`endif
    always @(posedge clk) begin
        if (acc && we) begin
            if (wstrb[0]) mem[addr][7:0]   <= wdata[7:0];
            if (wstrb[1]) mem[addr][15:8]  <= wdata[15:8];
            if (wstrb[2]) mem[addr][23:16] <= wdata[23:16];
            if (wstrb[3]) mem[addr][31:24] <= wdata[31:24];
        end
        q <= mem[addr];
    end

    // ---- latency pipeline (stage 0 lines up with the RAM output register)
    reg [NP-1:0] vpipe [0:LATENCY-1];
    reg [31:0]   dpipe [0:LATENCY-1];
    always @(posedge clk) begin : p_pipe
        integer k;
        vpipe[0] <= (rst || !(acc && !we)) ? {NP{1'b0}} : grant;
        for (k = 1; k < LATENCY; k = k + 1) begin
            vpipe[k] <= rst ? {NP{1'b0}} : vpipe[k-1];
            dpipe[k] <= (k == 1) ? q : dpipe[k-1];
        end
    end
    assign resp_valid = vpipe[LATENCY-1];
    assign resp_rdata = (LATENCY == 1) ? q : dpipe[LATENCY-1];
endmodule

`default_nettype wire
