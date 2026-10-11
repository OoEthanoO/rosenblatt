// rb_lsu.v: the global-memory load/store unit of one core.
//
// It executes one warp-wide global load or store at a time; meanwhile the
// other warps keep computing (and using shared memory), which is how a GPU
// hides memory latency.
//
// Coalescing: lanes are served lowest-first, and every waiting lane whose
// address falls in the same 32-bit word as the chosen lane rides along on the
// same request. 8 lanes reading 8 consecutive bytes cost 2 requests; 8 lanes
// reading one address (broadcast) cost 1; 8 scattered bytes cost 8.
// Requests are pipelined (one per cycle when the memory grants it); responses
// return in order and a small FIFO remembers which lanes each one feeds.
//
// When several lanes store to the same byte, the highest lane wins.
`default_nettype none

module rb_lsu #(
    parameter W       = 8,   // lanes (power of two, >= 2)
    parameter WB      = 2,   // warp-id bits
    parameter GMEM_AW = 16   // global memory size, log2(words)
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
    localparam S_IDLE = 2'd0, S_BUSY = 2'd1, S_DONE = 2'd2;

    reg  [1:0]      state;
    reg  [5:0]      op_q;
    reg  [WB-1:0]   warp_q;
    reg  [3:0]      rd_q;
    reg  [W-1:0]    mask_q;
    reg  [W*32-1:0] addr_q;
    reg  [W*32-1:0] sdata_q;
    reg  [W-1:0]    pending;   // lanes not yet sent to memory
    reg  [W*32-1:0] result;

    wire       is_store = op_q[2];
    wire [1:0] size     = op_q[1:0];   // 00 word, 01 signed byte, 10 unsigned byte

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

    // ---- the lowest waiting lane leads the next request
    reg [LB-1:0] lead;
    always @* begin : p_lead
        integer i;
        lead = {LB{1'b0}};
        for (i = W - 1; i >= 0; i = i - 1)
            if (pending[i]) lead = i;
    end
    wire [GMEM_AW-1:0] lead_word = addr_q[lead*32+2 +: GMEM_AW];

    reg [W-1:0] served;   // lanes served by this request
    always @* begin : p_cover
        integer i;
        for (i = 0; i < W; i = i + 1)
            served[i] = pending[i] && (addr_q[i*32+2 +: GMEM_AW] == lead_word);
    end

    // ---- merge the store data of all covered lanes into one word
    always @* begin : p_merge
        integer i;
        gm_req_wdata = 32'd0;
        gm_req_wstrb = 4'd0;
        for (i = 0; i < W; i = i + 1) begin
            if (served[i]) begin
                if (size == 2'b00) begin
                    gm_req_wdata = sdata_q[i*32 +: 32];
                    gm_req_wstrb = 4'hF;
                end else begin
                    gm_req_wdata[addr_q[i*32 +: 2]*8 +: 8] = sdata_q[i*32 +: 8];
                    gm_req_wstrb[addr_q[i*32 +: 2]]        = 1'b1;
                end
            end
        end
    end

    assign gm_req_valid = (state == S_BUSY) && (|pending);
    assign gm_req_we    = is_store;
    assign gm_req_addr  = lead_word;
    wire   fire         = gm_req_valid && gm_req_ready;

    // ---- which lanes each outstanding load request feeds (responses are in order)
    reg  [W-1:0]  fifo [0:W-1];
    reg  [LB-1:0] f_wp, f_rp;
    reg  [LB:0]   f_count;
    wire          f_push = fire && !is_store;
    wire          f_pop  = gm_resp_valid;
    wire [W-1:0]  f_head = fifo[f_rp];

    wire [W-1:0] pending_next = fire ? (pending & ~served) : pending;
    wire [LB:0]  count_next   = f_count + f_push - f_pop;
    wire         finishing    = (state == S_BUSY) && (pending_next == {W{1'b0}}) &&
                                (count_next == {(LB+1){1'b0}});

    // a new instruction may start in the cycle the previous one is written back
    assign ready = (state == S_IDLE) || (state == S_DONE && wb_ack);

    always @(posedge clk) begin : p_ctrl
        integer i;
        if (rst) begin
            state   <= S_IDLE;
            pending <= {W{1'b0}};
            f_wp    <= {LB{1'b0}};
            f_rp    <= {LB{1'b0}};
            f_count <= {(LB+1){1'b0}};
        end else begin
            if (state == S_BUSY) begin
                pending <= pending_next;
                if (finishing) state <= S_DONE;
            end else if (state == S_DONE && wb_ack) begin
                state <= S_IDLE;
            end
            if (start && ready) begin
                op_q    <= op;
                warp_q  <= warp;
                rd_q    <= rd;
                mask_q  <= mask;
                addr_q  <= addr;
                sdata_q <= sdata;
                pending <= mask;
                state   <= S_BUSY;
            end
            if (f_push) begin
                fifo[f_wp] <= served;
                f_wp       <= f_wp + 1'b1;
            end
            if (f_pop) f_rp <= f_rp + 1'b1;
            f_count <= count_next;
        end
        if (f_pop)
            for (i = 0; i < W; i = i + 1)
                if (f_head[i]) result[i*32 +: 32] <= extract(gm_resp_rdata, addr_q[i*32 +: 2], size);
    end

    assign idle      = (state == S_IDLE);
    assign wb_valid  = (state == S_DONE);
    assign wb_load   = !is_store;
    assign wb_warp   = warp_q;
    assign wb_rd     = rd_q;
    assign wb_mask   = mask_q;
    assign wb_data   = result;
    assign ev_gm_req = fire;
endmodule

`default_nettype wire
