// rb_core.v: one compute core (NVIDIA would call it an SM).
//
// A core runs one thread block at a time. The block's threads are grouped into
// warps of W threads; each warp shares one program counter and executes in
// lockstep on the core's W lanes (SIMT). The core holds up to NW warps and
// interleaves them, issuing one instruction per cycle from whichever warp is
// ready:
//
//   S0  schedule + fetch   pick a ready warp (round robin), read imem[pc]
//   S1  decode + reg read  every lane reads rs1, rs2, rd from its register file
//   S2  execute            ALUs compute; shared memory is accessed; the warp's
//                          pc / active mask / divergence stack are updated;
//                          global loads and stores go to the LSU
//   S3  write back         results (and shared-memory load data) are written
//
// A warp has at most one instruction between S0 and S2, so the pipeline needs
// no forwarding or hazard logic: a warp can issue every third cycle, and with
// three or more ready warps the core issues every cycle. While a warp waits for
// global memory the others keep going. That is latency hiding, the central
// idea of GPU throughput.
//
// Divergence: when the threads of a warp disagree on a branch, the warp runs
// the fall-through threads first and pushes {target, taken threads} on a
// per-warp stack. `ssy` pushes a reconvergence point and `sync` pops the stack
// (switching to deferred threads, or reconverging). Exited threads are tracked
// in xmask so they are never revived.
`default_nettype none
`include "rb_isa.vh"

module rb_core #(
    parameter CORE_ID     = 0,
    parameter W           = 8,    // lanes = threads per warp (power of two, >= 2)
    parameter NW          = 4,    // warp slots (power of two, >= 2)
    parameter IMEM_AW     = 10,
    parameter GMEM_AW     = 16,
    parameter SMEM_AW     = 10,
    parameter STACK_DEPTH = 8,
    parameter MUL_WIDTH   = 32,
    parameter ENABLE_DP4A = 1
) (
    input  wire                       clk,
    input  wire                       rst,
    // program loading (broadcast by the host)
    input  wire                       imem_we,
    input  wire [IMEM_AW-1:0]         imem_waddr,
    input  wire [31:0]                imem_wdata,
    // block dispatch
    input  wire                       start,
    input  wire [IMEM_AW-1:0]         entry,     // kernel start address
    input  wire [31:0]                ctaid,
    input  wire [31:0]                base_tid,  // ctaid * ntid
    input  wire [31:0]                ntid,
    input  wire [31:0]                nctaid,
    input  wire [8*32-1:0]            args,
    input  wire                       clear,     // kernel launch: zero counters and errors
    output wire                       idle,
    output reg  [2:0]                 errors,    // {sync underflow, stack overflow, illegal op}
    // global memory port
    output wire                       gm_req_valid,
    input  wire                       gm_req_ready,
    output wire                       gm_req_we,
    output wire [GMEM_AW-1:0]         gm_req_addr,
    output wire [31:0]                gm_req_wdata,
    output wire [3:0]                 gm_req_wstrb,
    input  wire                       gm_resp_valid,
    input  wire [31:0]                gm_resp_rdata,
    // performance counters
    output wire [`RB_NUM_PERF*32-1:0] perf
);
    localparam WB  = $clog2(NW);
    localparam LB  = $clog2(W);
    localparam SD  = STACK_DEPTH;
    localparam SPW = $clog2(SD + 1);
    localparam RFA = WB + 4;

    localparam WS_DONE = 3'd0, WS_READY = 3'd1, WS_BUSY = 3'd2, WS_MEM = 3'd3, WS_BAR = 3'd4;

    function [LB:0] popcount(input [W-1:0] m);
        integer k;
        begin
            popcount = 0;
            for (k = 0; k < W; k = k + 1) popcount = popcount + m[k];
        end
    endfunction

    // ================================================================ state
    reg                 running;
    reg  [31:0]         ctaid_q, base_q;
    reg  [31:0]         clock_cnt;

    reg  [2:0]          wstate [0:NW-1];
    reg  [IMEM_AW-1:0]  pc     [0:NW-1];
    reg  [W-1:0]        amask  [0:NW-1];  // threads currently executing
    reg  [W-1:0]        xmask  [0:NW-1];  // threads that have executed `ret`
    reg  [W-1:0]        rmask  [0:NW-1];  // lanes still owed by a bank-conflicted shared access
    reg  [SPW-1:0]      sp     [0:NW-1];
    reg  [NW-1:0]       unwind;           // next issue must pop the stack (no live threads)
    reg  [NW-1:0]       replaying;        // rmask is in use

    reg  [IMEM_AW-1:0]  stk_pc   [0:NW*SD-1];
    reg  [W-1:0]        stk_mask [0:NW*SD-1];

    // LSU interface (driven further down)
    wire                lsu_ready, lsu_idle;
    wire                lsu_wb_valid, lsu_wb_load;
    wire [WB-1:0]       lsu_wb_warp;
    wire [3:0]          lsu_wb_rd;
    wire [W-1:0]        lsu_wb_mask;
    wire [W*32-1:0]     lsu_wb_data;
    wire                ev_gm_req;

    // ============================================================ S0: issue
    reg  s1_valid, s2_valid, s3_valid;

    wire [NW-1:0] ready_vec, mem_wait_vec, bar_vec, done_vec;
    genvar gw;
    generate
        for (gw = 0; gw < NW; gw = gw + 1) begin : g_wstate
            assign ready_vec[gw]    = wstate[gw] == WS_READY;
            assign mem_wait_vec[gw] = wstate[gw] == WS_MEM;
            assign bar_vec[gw]      = wstate[gw] == WS_BAR;
            assign done_vec[gw]     = wstate[gw] == WS_DONE;
        end
    endgenerate

    reg  [WB-1:0] rr_last;
    reg  [WB-1:0] pick;
    reg           pick_ok;
    always @* begin : p_sched
        integer k;
        reg [WB-1:0] cand;
        pick    = rr_last;
        pick_ok = 1'b0;
        for (k = NW; k >= 1; k = k - 1) begin  // lowest offset after rr_last wins
            cand = rr_last + k[WB-1:0];
            if (ready_vec[cand]) begin
                pick    = cand;
                pick_ok = 1'b1;
            end
        end
    end

    // A finished global load needs the register-file write port, which S3
    // normally owns. If the pipeline is full, skip one issue slot to open a gap.
    wire hold_issue = lsu_wb_valid && lsu_wb_load && s1_valid && s2_valid && s3_valid;
    wire issue      = running && pick_ok && !hold_issue;

    reg [31:0] imem [0:(1<<IMEM_AW)-1];
    reg [31:0] imem_q;
    integer ii;
    initial for (ii = 0; ii < (1 << IMEM_AW); ii = ii + 1) imem[ii] = 32'd0;
    always @(posedge clk) begin
        if (imem_we) imem[imem_waddr] <= imem_wdata;
        imem_q <= imem[pc[pick]];
    end

    reg [WB-1:0]      s1_warp;
    reg [IMEM_AW-1:0] s1_pc;
    reg               s1_unwind;
    always @(posedge clk) begin
        s1_valid  <= rst ? 1'b0 : issue;
        s1_warp   <= pick;
        s1_pc     <= pc[pick];
        s1_unwind <= unwind[pick];
        if (rst) rr_last <= {WB{1'b0}};
        else if (issue) rr_last <= pick;
    end

    // ================================================== S1: decode + reg read
    wire [31:0] s1_instr = s1_unwind ? {`RB_OP_SYNC, 26'd0} : imem_q;
    wire [5:0]  s1_op    = s1_instr[31:26];
    wire [3:0]  s1_rd    = s1_instr[25:22];
    wire [3:0]  s1_rs1   = s1_instr[21:18];
    wire [3:0]  s1_rs2   = s1_instr[17:14];

    wire s1_is_alu  = (s1_op[5] == 1'b0) && !(s1_op[3:0] == `RB_FN_DP4A && !ENABLE_DP4A);
    wire s1_is_mem  = (s1_op[5:4] == 2'b10) && (s1_op[1:0] != 2'b11) &&
                      !(s1_op[2] && s1_op[1:0] == 2'b10);
    wire s1_is_li   = s1_op == `RB_OP_LI;
    wire s1_is_lui  = s1_op == `RB_OP_LUI;
    wire s1_is_s2r  = s1_op == `RB_OP_S2R;
    wire s1_is_bz   = s1_op == `RB_OP_BZ;
    wire s1_is_bnz  = s1_op == `RB_OP_BNZ;
    wire s1_is_ssy  = s1_op == `RB_OP_SSY;
    wire s1_is_sync = s1_op == `RB_OP_SYNC;
    wire s1_is_bar  = s1_op == `RB_OP_BAR;
    wire s1_is_ret  = s1_op == `RB_OP_RET;
    wire s1_illegal = !(s1_is_alu | s1_is_mem | s1_is_li | s1_is_lui | s1_is_s2r | s1_is_bz |
                        s1_is_bnz | s1_is_ssy | s1_is_sync | s1_is_bar | s1_is_ret);

    wire [RFA-1:0] rf_ra = {s1_warp, s1_rs1};
    wire [RFA-1:0] rf_rb = {s1_warp, s1_rs2};
    wire [RFA-1:0] rf_rc = {s1_warp, s1_rd};

    reg [WB-1:0]      s2_warp;
    reg [IMEM_AW-1:0] s2_pc;
    reg               s2_unwind;
    reg [5:0]         s2_op;
    reg [3:0]         s2_rd;
    reg [31:0]        s2_imm18, s2_imm22;
    reg s2_alu, s2_imm, s2_mem, s2_li, s2_lui, s2_s2r, s2_bz, s2_bnz;
    reg s2_ssy, s2_sync, s2_bar, s2_ret, s2_illegal, s2_writes;
    always @(posedge clk) begin
        s2_valid   <= rst ? 1'b0 : s1_valid;
        s2_warp    <= s1_warp;
        s2_pc      <= s1_pc;
        s2_unwind  <= s1_unwind;
        s2_op      <= s1_op;
        s2_rd      <= s1_rd;
        s2_imm18   <= {{14{s1_instr[17]}}, s1_instr[17:0]};
        s2_imm22   <= {{10{s1_instr[21]}}, s1_instr[21:0]};
        s2_alu     <= s1_is_alu;
        s2_imm     <= s1_op[4];
        s2_mem     <= s1_is_mem;
        s2_li      <= s1_is_li;
        s2_lui     <= s1_is_lui;
        s2_s2r     <= s1_is_s2r;
        s2_bz      <= s1_is_bz;
        s2_bnz     <= s1_is_bnz;
        s2_ssy     <= s1_is_ssy;
        s2_sync    <= s1_is_sync;
        s2_bar     <= s1_is_bar;
        s2_ret     <= s1_is_ret;
        s2_illegal <= s1_illegal;
        s2_writes  <= (s1_is_alu | s1_is_li | s1_is_lui | s1_is_s2r) && s1_rd != 4'd0;
    end

    // ======================================================= S2: execute
    wire [W-1:0]    M = amask[s2_warp];
    wire [W-1:0]    cond;       // per lane: rs1 != 0
    wire [W*32-1:0] ex_result;  // per lane: value for rd
    wire [W*32-1:0] mem_addr;   // per lane: rs1 + imm
    wire [W*32-1:0] st_data;    // per lane: rd (store data)

    // ---- register-file write port (S3): instruction results, else a finished global load
    reg             s3_alu_wr, s3_smem_wr;
    reg [WB-1:0]    s3_warp;
    reg [3:0]       s3_rd;
    reg [W-1:0]     s3_wmask;
    reg [W*32-1:0]  s3_data;
    wire [W*32-1:0] smem_rdata;

    wire            s3_writes   = s3_alu_wr || s3_smem_wr;
    wire            lsu_wb_take = lsu_wb_valid && (!lsu_wb_load || !s3_writes);
    wire            rf_lsu_we   = lsu_wb_take && lsu_wb_load && lsu_wb_rd != 4'd0;
    wire [RFA-1:0]  rf_wa       = s3_writes ? {s3_warp, s3_rd} : {lsu_wb_warp, lsu_wb_rd};
    wire [W*32-1:0] rf_wd       = s3_smem_wr ? smem_rdata : (s3_alu_wr ? s3_data : lsu_wb_data);
    wire [W-1:0]    rf_wmask    = s3_writes ? s3_wmask : (rf_lsu_we ? lsu_wb_mask : {W{1'b0}});

    wire [4:0] sr_sel = s2_imm18[4:0];

    genvar gl;
    generate
        for (gl = 0; gl < W; gl = gl + 1) begin : g_lane
            localparam [LB-1:0] LANE = gl;
            wire [31:0] qa, qb, qc, alu_y;
            rb_regfile #(.AW(RFA)) u_rf (
                .clk(clk), .ra(rf_ra), .rb(rf_rb), .rc(rf_rc),
                .qa(qa), .qb(qb), .qc(qc),
                .we(rf_wmask[gl]), .wa(rf_wa), .wd(rf_wd[gl*32 +: 32])
            );
            rb_alu #(.MUL_WIDTH(MUL_WIDTH), .ENABLE_DP4A(ENABLE_DP4A)) u_alu (
                .fn(s2_op[3:0]), .a(qa), .b(s2_imm ? s2_imm18 : qb), .c(qc), .y(alu_y)
            );
            wire [31:0] tid = {{(32-WB-LB){1'b0}}, s2_warp, LANE};
            reg  [31:0] srv;
            always @* begin
                case (sr_sel)
                    `RB_SR_TID:    srv = tid;
                    `RB_SR_NTID:   srv = ntid;
                    `RB_SR_CTAID:  srv = ctaid_q;
                    `RB_SR_NCTAID: srv = nctaid;
                    `RB_SR_LANEID: srv = {{(32-LB){1'b0}}, LANE};
                    `RB_SR_WARPID: srv = {{(32-WB){1'b0}}, s2_warp};
                    `RB_SR_COREID: srv = CORE_ID;
                    `RB_SR_CLOCK:  srv = clock_cnt;
                    `RB_SR_GTID:   srv = base_q + tid;
                    default:       srv = (sr_sel[4:3] == 2'b01) ? args[sr_sel[2:0]*32 +: 32] : 32'd0;
                endcase
            end
            assign ex_result[gl*32 +: 32] = s2_alu ? alu_y :
                                            s2_li  ? s2_imm22 :
                                            s2_lui ? {s2_imm22[21:0], 10'd0} : srv;
            assign cond[gl]              = qa != 32'd0;
            assign mem_addr[gl*32 +: 32] = qa + s2_imm18;
            assign st_data[gl*32 +: 32]  = qc;
        end
    endgenerate

    // ---- memory instructions
    wire         is_shared  = s2_mem && s2_op[3];
    wire         is_global  = s2_mem && !s2_op[3];
    wire [W-1:0] smem_lanes = replaying[s2_warp] ? rmask[s2_warp] : M;
    wire [W-1:0] smem_cov;
    rb_smem #(.W(W), .SMEM_AW(SMEM_AW)) u_smem (
        .clk(clk), .access(s2_valid && is_shared), .store(s2_op[2]), .size(s2_op[1:0]),
        .lanes(smem_lanes), .addr(mem_addr), .wdata(st_data),
        .covered(smem_cov), .rdata(smem_rdata)
    );
    wire [W-1:0] smem_rest = smem_lanes & ~smem_cov;   // lanes left for a replay

    wire lsu_start = s2_valid && is_global && lsu_ready;
    wire g_replay  = s2_valid && is_global && !lsu_ready;          // LSU busy: retry
    wire s_replay  = s2_valid && is_shared && smem_rest != {W{1'b0}};  // bank conflict
    wire replay    = g_replay || s_replay;

    // ---- control flow for the warp in S2
    wire [W-1:0]  br_taken  = M & (s2_bnz ? cond : ~cond);
    wire [W-1:0]  br_fall   = M & ~br_taken;
    wire          is_branch = s2_bz | s2_bnz;
    wire          divergent = is_branch && (br_taken != {W{1'b0}}) && (br_fall != {W{1'b0}});

    wire [SPW-1:0]     spv       = sp[s2_warp];
    wire               stk_empty = spv == {SPW{1'b0}};
    wire               stk_full  = spv == SD;
    wire [31:0]        top_idx   = s2_warp * SD + spv - 1;
    wire [31:0]        push_idx  = s2_warp * SD + spv;
    wire [IMEM_AW-1:0] top_pc    = stk_pc[top_idx];
    wire [W-1:0]       pop_mask  = stk_mask[top_idx] & ~xmask[s2_warp];

    wire [IMEM_AW-1:0] target  = s2_imm18[IMEM_AW-1:0];
    wire [IMEM_AW-1:0] pc_next = s2_pc + 1'b1;

    wire         want_push = s2_valid && (s2_ssy || divergent);
    wire         do_push   = want_push && !stk_full;
    wire [W-1:0] push_mask = s2_ssy ? M : br_taken;

    wire all_done    = &done_vec;
    wire bar_release = (&(bar_vec | done_vec)) && (|bar_vec);

    // lane mask of warp k for a block of `ntid` threads
    function [W-1:0] warp_lanes(input integer k);
        integer j;
        begin
            for (j = 0; j < W; j = j + 1) warp_lanes[j] = (k * W + j) < ntid;
        end
    endfunction

    // ================================================== warp state updates
    always @(posedge clk) begin : p_warps
        integer w;
        if (rst) begin
            running <= 1'b0;
            for (w = 0; w < NW; w = w + 1) begin
                wstate[w]    <= WS_DONE;
                sp[w]        <= {SPW{1'b0}};
                unwind[w]    <= 1'b0;
                replaying[w] <= 1'b0;
            end
        end else if (!running) begin
            if (start) begin
                running <= 1'b1;
                ctaid_q <= ctaid;
                base_q  <= base_tid;
                for (w = 0; w < NW; w = w + 1) begin
                    pc[w]        <= entry;
                    amask[w]     <= warp_lanes(w);
                    xmask[w]     <= {W{1'b0}};
                    sp[w]        <= {SPW{1'b0}};
                    unwind[w]    <= 1'b0;
                    replaying[w] <= 1'b0;
                    wstate[w]    <= (w * W < ntid) ? WS_READY : WS_DONE;
                end
            end
        end else begin
            if (all_done && lsu_idle && !s1_valid && !s2_valid && !s3_valid) running <= 1'b0;

            for (w = 0; w < NW; w = w + 1) begin
                if (issue && pick == w) wstate[w] <= WS_BUSY;
                if (bar_release && wstate[w] == WS_BAR) wstate[w] <= WS_READY;
                if (lsu_wb_take && lsu_wb_warp == w) wstate[w] <= WS_READY;

                if (s2_valid && s2_warp == w) begin
                    unwind[w] <= 1'b0;
                    wstate[w] <= WS_READY;
                    pc[w]     <= pc_next;
                    if (is_global) begin
                        if (g_replay) pc[w] <= s2_pc;
                        else          wstate[w] <= WS_MEM;
                    end else if (is_shared) begin
                        replaying[w] <= s_replay;
                        rmask[w]     <= smem_rest;
                        if (s_replay) pc[w] <= s2_pc;
                    end else if (is_branch) begin
                        if (br_taken == {W{1'b0}})     pc[w] <= pc_next;
                        else if (br_fall == {W{1'b0}}) pc[w] <= target;
                        else                           amask[w] <= br_fall;  // taken threads deferred
                    end else if (s2_sync) begin
                        if (!stk_empty) begin
                            sp[w]    <= spv - 1'b1;
                            pc[w]    <= top_pc;
                            amask[w] <= pop_mask;
                            if (pop_mask == {W{1'b0}}) begin
                                if (spv == 1) wstate[w] <= WS_DONE;
                                else          unwind[w] <= 1'b1;
                            end
                        end
                    end else if (s2_bar) begin
                        wstate[w] <= WS_BAR;
                    end else if (s2_ret) begin
                        xmask[w] <= xmask[w] | M;
                        amask[w] <= {W{1'b0}};
                        if (stk_empty) wstate[w] <= WS_DONE;
                        else           unwind[w] <= 1'b1;
                    end
                    if (do_push) sp[w] <= spv + 1'b1;
                end
            end

            if (do_push) begin
                stk_pc[push_idx]   <= target;
                stk_mask[push_idx] <= push_mask;
            end
        end
    end

    // ======================================================== S3 registers
    always @(posedge clk) begin
        s3_valid   <= rst ? 1'b0 : s2_valid;
        s3_alu_wr  <= !rst && s2_valid && s2_writes;
        s3_smem_wr <= !rst && s2_valid && is_shared && !s2_op[2] && s2_rd != 4'd0;
        s3_warp    <= s2_warp;
        s3_rd      <= s2_rd;
        s3_wmask   <= is_shared ? smem_cov : M;
        s3_data    <= ex_result;
    end

    // ============================================================ errors
    always @(posedge clk) begin
        if (rst || clear) errors <= 3'd0;
        else if (running && s2_valid) begin
            if (s2_illegal)            errors[0] <= 1'b1;
            if (want_push && stk_full) errors[1] <= 1'b1;
            if (s2_sync && stk_empty)  errors[2] <= 1'b1;
        end
    end

    // ============================================================== LSU
    rb_lsu #(.W(W), .WB(WB), .GMEM_AW(GMEM_AW)) u_lsu (
        .clk(clk), .rst(rst),
        .start(lsu_start), .op(s2_op), .warp(s2_warp), .rd(s2_rd), .mask(M),
        .addr(mem_addr), .sdata(st_data), .ready(lsu_ready), .idle(lsu_idle),
        .wb_valid(lsu_wb_valid), .wb_load(lsu_wb_load), .wb_warp(lsu_wb_warp),
        .wb_rd(lsu_wb_rd), .wb_mask(lsu_wb_mask), .wb_data(lsu_wb_data), .wb_ack(lsu_wb_take),
        .gm_req_valid(gm_req_valid), .gm_req_ready(gm_req_ready), .gm_req_we(gm_req_we),
        .gm_req_addr(gm_req_addr), .gm_req_wdata(gm_req_wdata), .gm_req_wstrb(gm_req_wstrb),
        .gm_resp_valid(gm_resp_valid), .gm_resp_rdata(gm_resp_rdata),
        .ev_gm_req(ev_gm_req)
    );

    // ============================================ performance counters
    reg [31:0] cnt [0:`RB_NUM_PERF-1];
    wire committed = s2_valid && !s2_unwind && !replay;
    always @(posedge clk) begin : p_perf
        integer k;
        if (rst || clear) begin
            for (k = 0; k < `RB_NUM_PERF; k = k + 1) cnt[k] <= 32'd0;
        end else begin
            if (running)                            cnt[`RB_PERF_ACTIVE]      <= cnt[`RB_PERF_ACTIVE] + 1;
            if (committed)                          cnt[`RB_PERF_ISSUED]      <= cnt[`RB_PERF_ISSUED] + 1;
            if (committed)                          cnt[`RB_PERF_LANE_OPS]    <= cnt[`RB_PERF_LANE_OPS] + popcount(M);
            if (running && !issue)                  cnt[`RB_PERF_IDLE]        <= cnt[`RB_PERF_IDLE] + 1;
            if (running && !issue && |mem_wait_vec) cnt[`RB_PERF_IDLE_MEM]    <= cnt[`RB_PERF_IDLE_MEM] + 1;
            if (replay)                             cnt[`RB_PERF_REPLAYS]     <= cnt[`RB_PERF_REPLAYS] + 1;
            if (lsu_start)                          cnt[`RB_PERF_GMEM_INSTR]  <= cnt[`RB_PERF_GMEM_INSTR] + 1;
            if (ev_gm_req)                          cnt[`RB_PERF_GMEM_REQS]   <= cnt[`RB_PERF_GMEM_REQS] + 1;
            if (lsu_start)                          cnt[`RB_PERF_GMEM_LANES]  <= cnt[`RB_PERF_GMEM_LANES] + popcount(M);
            if (committed && is_shared)             cnt[`RB_PERF_SMEM_INSTR]  <= cnt[`RB_PERF_SMEM_INSTR] + 1;
            if (s2_valid && is_shared)              cnt[`RB_PERF_SMEM_CYCLES] <= cnt[`RB_PERF_SMEM_CYCLES] + 1;
            if (!lsu_idle)                          cnt[`RB_PERF_LSU_BUSY]    <= cnt[`RB_PERF_LSU_BUSY] + 1;
            if (s2_valid && divergent)              cnt[`RB_PERF_DIVERGE]     <= cnt[`RB_PERF_DIVERGE] + 1;
            if (start && !running)                  cnt[`RB_PERF_BLOCKS]      <= cnt[`RB_PERF_BLOCKS] + 1;
        end
    end

    genvar gp;
    generate
        for (gp = 0; gp < `RB_NUM_PERF; gp = gp + 1) begin : g_perf
            assign perf[gp*32 +: 32] = cnt[gp];
        end
    endgenerate

    always @(posedge clk) clock_cnt <= rst ? 32'd0 : clock_cnt + 1;

    assign idle = !running;
endmodule

`default_nettype wire
