// rb_lsu.v: the global-memory load/store unit of one core.
//
// Global loads and stores of all warps of the core pass through here; while
// they are in flight the warps that issued them wait and the other warps keep
// computing, which is how a GPU hides memory latency.
//
// Two stages:
//   issue      takes one warp instruction and sends its requests, one per
//              cycle (when the memory grants it). Coalescing: lanes are served
//              lowest-first, and every waiting lane whose address falls in the
//              same 32-bit word as the chosen lane rides along on the same
//              request. 8 lanes reading 8 consecutive bytes cost 2 requests;
//              8 lanes reading one address (broadcast) cost 1; 8 scattered
//              bytes cost 8.
//   in flight  up to DEPTH instructions whose requests are all sent wait here
//              for their responses. The issue stage is free for the next
//              instruction as soon as the last request of the previous one is
//              sent, so several loads can be in flight at once
//              (memory-level parallelism). Responses come back in order; a
//              small FIFO records which slot and lanes each one feeds.
// Instructions complete (write back) in the order they were accepted.
//
// When several lanes store to the same byte, the highest lane wins.
`default_nettype none

module rb_lsu #(
    parameter W       = 8,   // lanes (power of two, >= 2)
    parameter WB      = 2,   // warp-id bits
    parameter GMEM_AW = 16,  // global memory size, log2(words)
    parameter DEPTH   = 4    // instructions in flight (power of two, >= 1)
) (
    input  wire               clk,
    input  wire               rst,
    // instruction handed over by the pipeline (accepted when `ready`)
    input  wire               start,
    input  wire [5:0]         op,
    input  wire [WB-1:0]      warp,
    input  wire [3:0]         rd,
    input  wire [W-1:0]       mask,
    input  wire [W*32-1:0]    addr,
    input  wire [W*32-1:0]    sdata,
    output wire               ready,
    output wire               idle,
    // completion: loads write wb_data into rd of the lanes in wb_mask
    output wire               wb_valid,
    output wire               wb_load,
    output wire [WB-1:0]      wb_warp,
    output wire [3:0]         wb_rd,
    output wire [W-1:0]       wb_mask,
    output wire [W*32-1:0]    wb_data,
    input  wire               wb_ack,
    // global memory port
    output wire               gm_req_valid,
    input  wire               gm_req_ready,
    output wire               gm_req_we,
    output wire [GMEM_AW-1:0] gm_req_addr,
    output reg  [31:0]        gm_req_wdata,
    output reg  [3:0]         gm_req_wstrb,
    input  wire               gm_resp_valid,
    input  wire [31:0]        gm_resp_rdata,
    // performance event: a request was accepted by the memory
    output wire               ev_gm_req
);
    localparam LB = $clog2(W);
    localparam QB = (DEPTH > 1) ? $clog2(DEPTH) : 1;   // slot index bits
    localparam FD = DEPTH * W;                         // response FIFO depth
    localparam FB = $clog2(FD) + 1;

    function [31:0] extract(input [31:0] w, input [1:0] off, input [1:0] sz);
        reg [7:0] b;
        begin
            case (off)
                2'd0: b = w[7:0];
                2'd1: b = w[15:8];
                2'd2: b = w[23:16];
                default: b = w[31:24];
            endcase
            case (sz)
                2'b00:   extract = w;
                2'b01:   extract = {{24{b[7]}}, b};
                default: extract = {24'd0, b};
            endcase
        end
    endfunction

    // ================================================================ issue
    reg             iu_busy;
    reg  [5:0]      iu_op;
    reg  [W-1:0]    iu_pending;   // lanes not yet sent
    reg  [W*32-1:0] iu_addr;
    reg  [W*32-1:0] iu_sdata;
    reg  [QB-1:0]   iu_slot;      // in-flight slot this instruction belongs to

    wire       iu_store = iu_op[2];
    wire [1:0] iu_size  = iu_op[1:0];   // 00 word, 01 signed byte, 10 unsigned byte

    // the lowest waiting lane leads the next request (constant-index loops only)
    reg [GMEM_AW-1:0] lead_word;
    always @* begin : p_lead
        integer i;
        lead_word = {GMEM_AW{1'b0}};
        for (i = W - 1; i >= 0; i = i - 1)
            if (iu_pending[i]) lead_word = iu_addr[i*32+2 +: GMEM_AW];
    end

    reg [W-1:0] served;   // lanes served by this request
    always @* begin : p_cover
        integer i;
        for (i = 0; i < W; i = i + 1)
            served[i] = iu_pending[i] && (iu_addr[i*32+2 +: GMEM_AW] == lead_word);
    end

    // merge the store data of all served lanes into one word
    always @* begin : p_merge
        integer i;
        gm_req_wdata = 32'd0;
        gm_req_wstrb = 4'd0;
        for (i = 0; i < W; i = i + 1) begin
            if (served[i]) begin
                if (iu_size == 2'b00) begin
                    gm_req_wdata = iu_sdata[i*32 +: 32];
                    gm_req_wstrb = 4'hF;
                end else begin
                    case (iu_addr[i*32 +: 2])
                        2'd0: begin gm_req_wdata[7:0]   = iu_sdata[i*32 +: 8]; gm_req_wstrb[0] = 1'b1; end
                        2'd1: begin gm_req_wdata[15:8]  = iu_sdata[i*32 +: 8]; gm_req_wstrb[1] = 1'b1; end
                        2'd2: begin gm_req_wdata[23:16] = iu_sdata[i*32 +: 8]; gm_req_wstrb[2] = 1'b1; end
                        default: begin gm_req_wdata[31:24] = iu_sdata[i*32 +: 8]; gm_req_wstrb[3] = 1'b1; end
                    endcase
                end
            end
        end
    end

    assign gm_req_valid = iu_busy && (|iu_pending);
    assign gm_req_we    = iu_store;
    assign gm_req_addr  = lead_word;
    wire   fire         = gm_req_valid && gm_req_ready;
    wire   last_fire    = fire && ((iu_pending & ~served) == {W{1'b0}});

    // ============================================================ in flight
    reg  [DEPTH-1:0]    q_valid, q_issued, q_load;
    reg  [WB-1:0]       q_warp   [0:DEPTH-1];
    reg  [3:0]          q_rd     [0:DEPTH-1];
    reg  [W-1:0]        q_mask   [0:DEPTH-1];
    reg  [1:0]          q_size   [0:DEPTH-1];
    reg  [LB:0]         q_count  [0:DEPTH-1];   // responses still to come
    // per-lane byte offsets and results live in small per-lane memories (g_lane)
    reg  [QB-1:0]       head, tail;
    reg  [QB:0]         used;

    // response FIFO: which slot and lanes each outstanding load request feeds
    reg  [QB+W-1:0]     rfifo [0:FD-1];
    reg  [FB-2:0]       r_wp, r_rp;
    wire [QB+W-1:0]     r_head  = rfifo[r_rp];
    wire [QB-1:0]       r_slot  = r_head[QB+W-1:W];
    wire [W-1:0]        r_lanes = r_head[W-1:0];

    assign ready = !iu_busy && (used != DEPTH);
    wire accept  = start && ready;

    wire h_done  = q_valid[head] && q_issued[head] && q_count[head] == 0 &&
                   !(gm_resp_valid && r_slot == head);
    assign wb_valid = h_done;
    assign wb_load  = q_load[head];
    assign wb_warp  = q_warp[head];
    assign wb_rd    = q_rd[head];
    assign wb_mask  = q_mask[head];
    wire   retire   = h_done && wb_ack;

    always @(posedge clk) begin : p_ctrl
        integer s;
        if (rst) begin
            iu_busy  <= 1'b0;
            q_valid  <= {DEPTH{1'b0}};
            head     <= {QB{1'b0}};
            tail     <= {QB{1'b0}};
            used     <= {(QB+1){1'b0}};
            r_wp     <= {(FB-1){1'b0}};
            r_rp     <= {(FB-1){1'b0}};
        end else begin
            // ---- accept a new instruction into the issue stage and a slot
            if (accept) begin
                iu_busy    <= 1'b1;
                iu_op      <= op;
                iu_pending <= mask;
                iu_addr    <= addr;
                iu_sdata   <= sdata;
                iu_slot    <= tail;
                q_valid[tail]  <= 1'b1;
                q_issued[tail] <= 1'b0;
                q_load[tail]   <= !op[2];
                q_warp[tail]   <= warp;
                q_rd[tail]     <= rd;
                q_mask[tail]   <= mask;
                q_size[tail]   <= op[1:0];
                tail <= tail + 1'b1;
            end
            // ---- send requests
            if (fire) begin
                iu_pending <= iu_pending & ~served;
                if (!iu_store) begin
                    rfifo[r_wp] <= {iu_slot, served};
                    r_wp        <= r_wp + 1'b1;
                end
                if (last_fire) begin
                    iu_busy           <= 1'b0;
                    q_issued[iu_slot] <= 1'b1;
                end
            end
            // ---- count responses in and out per slot
            for (s = 0; s < DEPTH; s = s + 1) begin
                if (accept && tail == s)
                    q_count[s] <= {(LB+1){1'b0}};
                else
                    q_count[s] <= q_count[s] + (fire && !iu_store && iu_slot == s)
                                             - (gm_resp_valid && r_slot == s);
            end
            if (gm_resp_valid) r_rp <= r_rp + 1'b1;
            // ---- retire the oldest instruction
            if (retire) begin
                q_valid[head] <= 1'b0;
                head          <= head + 1'b1;
            end
            used <= used + accept - retire;
        end
    end

    // ---- per lane: byte offset of each in-flight load, and its result word
    genvar gl;
    generate
        for (gl = 0; gl < W; gl = gl + 1) begin : g_lane
            reg [1:0]  off [0:DEPTH-1];
            reg [31:0] res [0:DEPTH-1];
            always @(posedge clk) begin
                if (accept) off[tail] <= addr[gl*32 +: 2];
                if (gm_resp_valid && r_lanes[gl])
                    res[r_slot] <= extract(gm_resp_rdata, off[r_slot], q_size[r_slot]);
            end
            assign wb_data[gl*32 +: 32] = res[head];
        end
    endgenerate

    assign idle      = !iu_busy && (used == 0);
    assign ev_gm_req = fire;

`ifndef SYNTHESIS
    initial begin
        if (DEPTH < 1 || (DEPTH & (DEPTH - 1)) != 0)
            $fatal(1, "rb_lsu: DEPTH must be a power of two");
    end
`endif
endmodule

`default_nettype wire
