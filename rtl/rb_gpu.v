// rb_gpu.v: the Rosenblatt GPU, top level.
//
//            host bus (testbench / UART bridge)
//                         |
//        +----------------+------------------+
//        |  control registers + dispatcher   |  launches a grid of blocks,
//        +---+--------------+-----------+----+  hands each idle core the next one
//            |              |           |
//        +---+----+    +----+---+       |
//        | core 0 |    | core 1 |  ...  |       each core: warps, lanes, LSU,
//        +---+----+    +----+---+       |       shared memory, imem copy
//            |              |           |
//        +---+--------------+-----------+----+
//        |   global memory (arbitrated)      |
//        +-----------------------------------+
//
// Host bus: word addressed, one request per handshake (valid/ready). Reads
// return later on rvalid/rdata, in order; only one read is outstanding at a
// time. Address map in sw/rosenblatt/isa.py (HOST_*_BASE, CSR).
`default_nettype none
`include "rb_isa.vh"

module rb_gpu #(
    parameter NUM_CORES     = 2,
    parameter NUM_LANES     = 8,
    parameter NUM_WARPS     = 4,
    parameter IMEM_AW       = 10,
    parameter GMEM_AW       = 16,
    parameter SMEM_AW       = 10,
    parameter STACK_DEPTH   = 8,
    parameter MUL_WIDTH     = 32,
    parameter ENABLE_DP4A   = 1,
    parameter GMEM_LATENCY  = 8,
    parameter GMEM_INTERVAL = 1,
    parameter LSU_DEPTH     = 4,
    parameter PARK          = 0
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        host_valid,
    output wire        host_ready,
    input  wire        host_we,
    input  wire [23:0] host_addr,
    input  wire [31:0] host_wdata,
    output reg         host_rvalid,
    output reg  [31:0] host_rdata,
    output wire        busy
);
    localparam NC = NUM_CORES;
    localparam NP = NC + 1;      // global memory ports: cores, then the host
    localparam HP = NC;          // host port index
    localparam CB = (NC > 1) ? $clog2(NC) : 1;

    // configuration readable by software (CSR HWCFG*)
    localparam [7:0] P_CORES = NUM_CORES,  P_WARPS = NUM_WARPS, P_LANES = NUM_LANES;
    localparam [7:0] P_SMEM  = SMEM_AW,    P_GMEM  = GMEM_AW,   P_IMEM  = IMEM_AW;
    localparam [7:0] P_MULW  = MUL_WIDTH,  P_STACK = STACK_DEPTH, P_LAT = GMEM_LATENCY;
    localparam [7:0] P_FEAT  = ENABLE_DP4A ? `RB_FEATURE_DP4A : 8'd0;
    localparam [7:0] P_LSUD  = LSU_DEPTH,  P_INTV  = GMEM_INTERVAL, P_PARK = PARK;

    // ================================================================ host bus
    wire h_gmem = host_addr[23:20] == 4'h0;
    wire h_imem = host_addr[23:20] == 4'h1;
    wire h_csr  = host_addr[23:20] == 4'h2;
    wire [11:0] csr_off = host_addr[11:0];

    reg  rd_pending;  // a global-memory read for the host is in flight

    wire [NP-1:0] gm_ready, gm_resp_valid, gm_accepted;
    wire [31:0]   gm_resp_rdata;

    wire hg_valid = host_valid && h_gmem && !rd_pending;
    assign host_ready = !rd_pending && (h_gmem ? gm_ready[HP] : 1'b1);
    wire h_fire = host_valid && host_ready;

    // ============================================================= CSRs
    reg         running, done;
    reg  [31:0] grid, ntid, entry;
    reg  [31:0] args [0:7];
    reg  [31:0] cycles, gmem_reqs;
    reg  [31:0] next_block, next_base;

    wire launch = h_fire && host_we && h_csr && csr_off == `RB_CSR_CTRL && host_wdata[0] && !running;

    wire [NC-1:0]                 core_idle;
    wire [NC*3-1:0]               core_err;
    wire [NC*`RB_NUM_PERF*32-1:0] core_perf;

    // lowest idle core gets the next block
    reg  [CB-1:0] free_core;
    reg           free_ok;
    always @* begin : p_free
        integer c;
        free_core = {CB{1'b0}};
        free_ok   = 1'b0;
        for (c = NC - 1; c >= 0; c = c - 1)
            if (core_idle[c]) begin
                free_core = c;
                free_ok   = 1'b1;
            end
    end
    wire blocks_left = next_block < grid;
    wire dispatch    = running && blocks_left && free_ok;
    wire finish      = running && !blocks_left && (&core_idle);

    integer a;
    always @(posedge clk) begin
        if (rst) begin
            running <= 1'b0;
            done    <= 1'b0;
            grid    <= 32'd0;
            ntid    <= 32'd0;
            entry   <= 32'd0;
            cycles  <= 32'd0;
            gmem_reqs <= 32'd0;
            next_block <= 32'd0;
            next_base  <= 32'd0;
            for (a = 0; a < 8; a = a + 1) args[a] <= 32'd0;
        end else begin
            if (h_fire && host_we && h_csr && !running) begin
                if (csr_off == `RB_CSR_GRID)  grid <= host_wdata;
                if (csr_off == `RB_CSR_BLOCK) ntid <= host_wdata;
                if (csr_off == `RB_CSR_ENTRY) entry <= host_wdata;
                if (csr_off >= `RB_CSR_ARG0 && csr_off < `RB_CSR_ARG0 + 8)
                    args[csr_off[2:0]] <= host_wdata;
            end
            if (launch) begin
                running    <= 1'b1;
                done       <= 1'b0;
                cycles     <= 32'd0;
                gmem_reqs  <= 32'd0;
                next_block <= 32'd0;
                next_base  <= 32'd0;
            end else if (running) begin
                cycles <= cycles + 1;
                if (|gm_accepted[NC-1:0]) gmem_reqs <= gmem_reqs + 1;
                if (dispatch) begin
                    next_block <= next_block + 1;
                    next_base  <= next_base + ntid;
                end
                if (finish) begin
                    running <= 1'b0;
                    done    <= 1'b1;
                end
            end
        end
    end
    assign busy = running;

    // ---- CSR / memory read responses
    reg [31:0] csr_rdata;
    always @* begin : p_csr
        integer c, k;
        c = 0;          // loop variables are assigned on every path,
        k = 0;          // otherwise Yosys infers latches for them
        csr_rdata = 32'd0;
        case (csr_off)
            `RB_CSR_CTRL:      csr_rdata = {29'd0, |core_err, done, running};
            `RB_CSR_GRID:      csr_rdata = grid;
            `RB_CSR_BLOCK:     csr_rdata = ntid;
            `RB_CSR_ENTRY:     csr_rdata = entry;
            `RB_CSR_CYCLES:    csr_rdata = cycles;
            `RB_CSR_GMEM_REQS: csr_rdata = gmem_reqs;
            `RB_CSR_ERRORS:    csr_rdata = core_err;
            `RB_CSR_HWCFG0:    csr_rdata = {`RB_HW_VERSION, P_CORES, P_WARPS, P_LANES};
            `RB_CSR_HWCFG1:    csr_rdata = {P_FEAT, P_SMEM, P_GMEM, P_IMEM};
            `RB_CSR_HWCFG2:    csr_rdata = {8'd0, P_MULW, P_STACK, P_LAT};
            `RB_CSR_HWCFG3:    csr_rdata = {8'd0, P_PARK, P_INTV, P_LSUD};
            default: begin
                for (k = 0; k < 8; k = k + 1)
                    if (csr_off == `RB_CSR_ARG0 + k) csr_rdata = args[k];
                // performance counters: constant indices only (no multiplier)
                for (c = 0; c < NC; c = c + 1)
                    for (k = 0; k < `RB_NUM_PERF; k = k + 1)
                        if (csr_off == `RB_CSR_PERF_BASE + c * `RB_PERF_STRIDE + k)
                            csr_rdata = core_perf[(c * `RB_NUM_PERF + k) * 32 +: 32];
            end
        endcase
    end

    always @(posedge clk) begin
        if (rst) begin
            rd_pending  <= 1'b0;
            host_rvalid <= 1'b0;
            host_rdata  <= 32'd0;
        end else begin
            host_rvalid <= 1'b0;
            if (h_fire && !host_we) begin
                if (h_gmem) rd_pending <= 1'b1;
                else begin
                    host_rvalid <= 1'b1;
                    host_rdata  <= h_csr ? csr_rdata : 32'd0;
                end
            end
            if (gm_resp_valid[HP]) begin
                rd_pending  <= 1'b0;
                host_rvalid <= 1'b1;
                host_rdata  <= gm_resp_rdata;
            end
        end
    end

    // ============================================================= cores
    wire imem_we = h_fire && host_we && h_imem && !running;

    wire [NP-1:0]         gm_valid, gm_we;
    wire [NP*GMEM_AW-1:0] gm_addr;
    wire [NP*32-1:0]      gm_wdata;
    wire [NP*4-1:0]       gm_wstrb;

    wire [8*32-1:0] args_flat = {args[7], args[6], args[5], args[4],
                                 args[3], args[2], args[1], args[0]};

    genvar gc;
    generate
        for (gc = 0; gc < NC; gc = gc + 1) begin : g_core
            rb_core #(
                .CORE_ID(gc), .W(NUM_LANES), .NW(NUM_WARPS), .IMEM_AW(IMEM_AW),
                .GMEM_AW(GMEM_AW), .SMEM_AW(SMEM_AW), .STACK_DEPTH(STACK_DEPTH),
                .MUL_WIDTH(MUL_WIDTH), .ENABLE_DP4A(ENABLE_DP4A),
                .LSU_DEPTH(LSU_DEPTH), .PARK(PARK)
            ) u_core (
                .clk(clk), .rst(rst),
                .imem_we(imem_we), .imem_waddr(host_addr[IMEM_AW-1:0]), .imem_wdata(host_wdata),
                .start(dispatch && free_core == gc), .entry(entry[IMEM_AW-1:0]), .ctaid(next_block), .base_tid(next_base),
                .ntid(ntid), .nctaid(grid), .args(args_flat), .clear(launch),
                .idle(core_idle[gc]), .errors(core_err[gc*3 +: 3]),
                .gm_req_valid(gm_valid[gc]), .gm_req_ready(gm_ready[gc]), .gm_req_we(gm_we[gc]),
                .gm_req_addr(gm_addr[gc*GMEM_AW +: GMEM_AW]), .gm_req_wdata(gm_wdata[gc*32 +: 32]),
                .gm_req_wstrb(gm_wstrb[gc*4 +: 4]),
                .gm_resp_valid(gm_resp_valid[gc]), .gm_resp_rdata(gm_resp_rdata),
                .perf(core_perf[gc*`RB_NUM_PERF*32 +: `RB_NUM_PERF*32])
            );
        end
    endgenerate

    // host port on the global memory
    assign gm_valid[HP]                    = hg_valid;
    assign gm_we[HP]                       = host_we;
    assign gm_addr[HP*GMEM_AW +: GMEM_AW]  = host_addr[GMEM_AW-1:0];
    assign gm_wdata[HP*32 +: 32]           = host_wdata;
    assign gm_wstrb[HP*4 +: 4]             = 4'hF;

    rb_gmem #(.NP(NP), .AW(GMEM_AW), .LATENCY(GMEM_LATENCY), .INTERVAL(GMEM_INTERVAL)) u_gmem (
        .clk(clk), .rst(rst),
        .req_valid(gm_valid), .req_ready(gm_ready), .req_we(gm_we), .req_addr(gm_addr),
        .req_wdata(gm_wdata), .req_wstrb(gm_wstrb),
        .resp_valid(gm_resp_valid), .resp_rdata(gm_resp_rdata), .accepted(gm_accepted)
    );
endmodule

`default_nettype wire
