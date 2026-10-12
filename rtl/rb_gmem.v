// rb_gmem.v: global memory, shared by every core and the host.
//
// One line of LINE bytes per request (4 by default, a single 32-bit word), at
// most one request every INTERVAL cycles for the whole device: this is the
// bandwidth bottleneck the profiling chapter is about. Reads return LATENCY
// cycles after they are accepted, in order; a round-robin arbiter picks among
// the requesting ports. Writes take effect when accepted (byte enables) and
// send no response.
//
// Wider lines model what real GPUs do (32-byte sectors): one request moves more
// bytes, which helps whenever the lanes of a warp touch neighbouring addresses.
//
// On an FPGA the array maps to block RAM; the extra latency stages model the
// DRAM a real GPU would talk to.
`default_nettype none

module rb_gmem #(
    parameter NP       = 3,   // ports: cores + host
    parameter AW       = 16,  // log2(words): total size
    parameter LINE     = 4,   // bytes per request: 4, 8, 16 or 32
    parameter LATENCY  = 8,   // >= 1
    parameter INTERVAL = 1    // >= 1
) (
    input  wire                  clk,
    input  wire                  rst,
    input  wire [NP-1:0]         req_valid,
    output wire [NP-1:0]         req_ready,
    input  wire [NP-1:0]         req_we,
    input  wire [NP*(AW-$clog2(LINE/4))-1:0] req_addr,   // line addresses
    input  wire [NP*LINE*8-1:0]  req_wdata,
    input  wire [NP*LINE-1:0]    req_wstrb,
    output wire [NP-1:0]         resp_valid,
    output wire [LINE*8-1:0]     resp_rdata,
    output wire [NP-1:0]         accepted     // which port was served this cycle
);
    localparam PB  = (NP > 1) ? $clog2(NP) : 1;
    localparam LAW = AW - $clog2(LINE / 4);   // line-address bits
    localparam LB  = LINE * 8;                // line bits

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

    wire            we    = req_we[sel];
    wire [LAW-1:0]  addr  = req_addr[sel*LAW +: LAW];
    wire [LB-1:0]   wdata = req_wdata[sel*LB +: LB];
    wire [LINE-1:0] wstrb = req_wstrb[sel*LINE +: LINE];

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
    reg [LB-1:0] mem [0:(1<<LAW)-1];
    reg [LB-1:0] q;
    integer ii;
`ifndef SYNTHESIS  // simulation only: FPGA block RAM powers up as zero
    initial for (ii = 0; ii < (1 << LAW); ii = ii + 1) mem[ii] = {LB{1'b0}};
`endif
    always @(posedge clk) begin : p_mem
        integer b;
        if (acc && we) begin
            for (b = 0; b < LINE; b = b + 1)
                if (wstrb[b]) mem[addr][b*8 +: 8] <= wdata[b*8 +: 8];
        end
        q <= mem[addr];
    end

    // ---- latency pipeline (stage 0 lines up with the RAM output register)
    reg [NP-1:0] vpipe [0:LATENCY-1];
    reg [LB-1:0] dpipe [0:LATENCY-1];
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
