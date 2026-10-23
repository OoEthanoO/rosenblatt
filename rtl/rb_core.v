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
// Global loads: by default a warp that issues one waits until its data is
// written back. With SCOREBOARD = 1 it keeps going: the load marks its
// destination register as pending, and only an instruction that reads or
// writes a pending register waits (it is dropped in S1, and the warp sleeps
// until those registers arrive). One warp can then have several loads in
// flight. Stores never stall the warp in that mode.
//
// Matrix unit (ENABLE_MMA = 1, needs ENABLE_DP4A): `mma ra, rb` multiplies a
// W x 4 int8 matrix A (row j = lane j's ra) by a 4 x W one B (column i = lane
// i's rb) into the warp's W x W int32 accumulator tile, which lives in the
// unit, not in the register file. The instruction leaves the pipeline in S2
// like any other: its operands are copied into one of the warp's MMA_SLOTS
// slots, and the unit then computes one row of the tile per cycle on the
// lanes' dot-product multipliers, which it borrows whenever the instruction
// in S2 is not a dp4a. The warp runs on meanwhile. Its mma and mmaz wait
// (WS_MMA) only while all its slots are taken, and the unit runs them in
// order. `mmar rd, j` copies row j of the tile into a register once the unit
// has finished every operation of the warp, and `mmaz` clears the tile.
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
    parameter GMEM_LINE   = 4,    // bytes per global-memory request
    parameter SMEM_AW     = 10,
    parameter STACK_DEPTH = 8,
    parameter MUL_WIDTH   = 32,
    parameter ENABLE_DP4A = 1,
    parameter LSU_DEPTH   = 4,    // global memory instructions in flight per core
    parameter PARK        = 0,    // 1: a warp that finds the LSU busy sleeps until it is free
    parameter SCOREBOARD  = 0,    // 1: a global load stalls only the instructions that use its register
    parameter ENABLE_MMA  = 0,    // 1: the matrix unit (mma, mmaz, mmar); needs ENABLE_DP4A
    parameter MMA_SLOTS   = 1     // mma/mmaz per warp in the matrix unit at once (power of two)
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
    output wire [GMEM_AW-$clog2(GMEM_LINE/4)-1:0] gm_req_addr,   // line address
    output wire [GMEM_LINE*8-1:0]     gm_req_wdata,
    output wire [GMEM_LINE-1:0]       gm_req_wstrb,
    input  wire                       gm_resp_valid,
    input  wire [GMEM_LINE*8-1:0]     gm_resp_rdata,
    // performance counters
    output wire [`RB_NUM_PERF*32-1:0] perf
);
    localparam WB  = $clog2(NW);
    localparam LB  = $clog2(W);
    localparam SD  = STACK_DEPTH;      // power of two
    localparam SDB = $clog2(SD);
    localparam SPW = SDB + 1;          // stack pointer 0..SD
    localparam MAXT = W * NW;          // threads per block the core can hold
    localparam RFA = WB + 4;

    localparam WS_DONE = 3'd0, WS_READY = 3'd1, WS_BUSY = 3'd2, WS_MEM = 3'd3, WS_BAR = 3'd4;
    localparam WS_LSU  = 3'd5;   // parked: its global memory instruction found the LSU busy
    localparam WS_DEP  = 3'd6;   // scoreboard: waits for registers that loads have not filled yet
    localparam WS_MMA  = 3'd7;   // waits for a free matrix-unit slot (mmar: for all its operations)
    localparam MMA     = ENABLE_MMA && ENABLE_DP4A;
    localparam MS      = MMA_SLOTS;
    localparam SB      = (MS > 1) ? $clog2(MS) : 1;   // slot number bits
    localparam MCB     = $clog2(MS + 1);              // a count of slots, 0..MS

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

    // scoreboard (SCOREBOARD = 1): registers of each warp that a load will still write
    reg  [15:0]         pend     [0:NW-1];
    reg  [15:0]         dep_need [0:NW-1];  // the registers a sleeping warp waits for

    // matrix unit (MMA): each warp has up to MS mma/mmaz in the unit, in a ring
    // of slots; slot s of warp w is entry w * MS + s of the slot memories
    reg  [MCB-1:0]      mq_cnt   [0:NW-1];  // waiting for the unit
    reg  [MCB-1:0]      mp_cnt   [0:NW-1];  // waiting, executing, or the last row not yet written
    reg  [SB-1:0]       mq_next  [0:NW-1];  // the slot of the next one to start
    reg  [NW-1:0]       mma_all;            // in WS_MMA for an mmar: waits until mp_cnt is 0

    function [WB+SB-1:0] mslot(input [WB-1:0] w, input [SB-1:0] s);
        mslot = (MS > 1) ? {w, s} : w;
    endfunction

    wire [NW-1:0]       mq_any;             // has an operation waiting for the unit
    wire [NW-1:0]       mp_any;             // has an operation in the unit
    wire [NW-1:0]       mp_free;            // has a free slot
    genvar gm;
    generate
        for (gm = 0; gm < NW; gm = gm + 1) begin : g_mslots
            assign mq_any[gm]  = mq_cnt[gm] != {MCB{1'b0}};
            assign mp_any[gm]  = mp_cnt[gm] != {MCB{1'b0}};
            assign mp_free[gm] = mp_cnt[gm] != MS;
        end
    endgenerate

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
            assign mem_wait_vec[gw] = wstate[gw] == WS_MEM || wstate[gw] == WS_LSU ||
                                      wstate[gw] == WS_DEP;
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
`ifndef SYNTHESIS  // simulation only: FPGA block RAM powers up as zero
    initial for (ii = 0; ii < (1 << IMEM_AW); ii = ii + 1) imem[ii] = 32'd0;
`endif
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
    wire s1_is_dp4ab = (s1_op == `RB_OP_DP4AB) && ENABLE_DP4A;   // dp4a, rs1 from one lane
    wire s1_is_mma   = (s1_op == `RB_OP_MMA)  && MMA;            // matrix unit
    wire s1_is_mmaz  = (s1_op == `RB_OP_MMAZ) && MMA;
    wire s1_is_mmar  = (s1_op == `RB_OP_MMAR) && MMA;
    wire s1_illegal = !(s1_is_alu | s1_is_mem | s1_is_li | s1_is_lui | s1_is_s2r | s1_is_bz |
                        s1_is_bnz | s1_is_ssy | s1_is_sync | s1_is_bar | s1_is_ret | s1_is_dp4ab |
                        s1_is_mma | s1_is_mmaz | s1_is_mmar);

    // registers the instruction reads or writes, checked against the scoreboard
    wire        s1_reads_rs1 = s1_is_alu | s1_is_dp4ab | s1_is_mem | s1_is_bz | s1_is_bnz | s1_is_mma;
    wire        s1_reads_rs2 = (s1_is_alu && !s1_op[4]) | s1_is_dp4ab | s1_is_mma;
    wire        s1_uses_rd   = s1_is_alu | s1_is_dp4ab | s1_is_mem | s1_is_li | s1_is_lui | s1_is_s2r |
                               s1_is_mmar;
    wire [15:0] s1_regs      = (s1_reads_rs1 ? 16'd1 << s1_rs1 : 16'd0) |
                               (s1_reads_rs2 ? 16'd1 << s1_rs2 : 16'd0) |
                               (s1_uses_rd   ? 16'd1 << s1_rd  : 16'd0);
    wire        s1_reg_wait  = SCOREBOARD && (pend[s1_warp] & s1_regs) != 16'd0;
    // an mma/mmaz waits for a free slot, an mmar until the unit is done with the warp
    wire        s1_mma_wait  = ((s1_is_mma | s1_is_mmaz) && !mp_free[s1_warp]) ||
                               (s1_is_mmar && mp_any[s1_warp]);
    wire        s1_hazard    = s1_valid && !s1_unwind && (s1_reg_wait || s1_mma_wait);

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
    reg s2_bcast;                     // dp4ab: rs1 comes from lane s2_lane
    reg [LB-1:0] s2_lane;             // dp4ab's lane, mmar's tile row
    reg s2_mma, s2_mmaz, s2_mmar;
    reg s2_dot;                       // dp4a or dp4ab: needs the dot-product path
    reg s2_ssy, s2_sync, s2_bar, s2_ret, s2_illegal, s2_writes;
    always @(posedge clk) begin
        s2_valid   <= rst ? 1'b0 : s1_valid && !s1_hazard;
        s2_warp    <= s1_warp;
        s2_pc      <= s1_pc;
        s2_unwind  <= s1_unwind;
        s2_op      <= s1_op;
        s2_rd      <= s1_rd;
        s2_imm18   <= {{14{s1_instr[17]}}, s1_instr[17:0]};
        s2_imm22   <= {{10{s1_instr[21]}}, s1_instr[21:0]};
        s2_alu     <= s1_is_alu | s1_is_dp4ab;
        s2_imm     <= s1_op[4] && !s1_is_dp4ab;
        s2_bcast   <= s1_is_dp4ab;
        s2_lane    <= s1_instr[LB-1:0];
        s2_mma     <= s1_is_mma;
        s2_mmaz    <= s1_is_mmaz;
        s2_mmar    <= s1_is_mmar;
        s2_dot     <= (s1_is_alu && s1_op[3:0] == `RB_FN_DP4A) | s1_is_dp4ab;
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
        s2_writes  <= (s1_is_alu | s1_is_dp4ab | s1_is_li | s1_is_lui | s1_is_s2r | s1_is_mmar) &&
                      s1_rd != 4'd0;
    end

    // ======================================================= S2: execute
    wire [W-1:0]    M = amask[s2_warp];
    wire [W-1:0]    cond;       // per lane: rs1 != 0
    wire [W*32-1:0] ex_result;  // per lane: value for rd (li, lui, s2r, mmar; ALU results come in S3)
    wire [W*32-1:0] alu_res;    // per lane: ALU result of the instruction now in S3
    wire [W*32-1:0] mem_addr;   // per lane: rs1 + imm
    wire [W*32-1:0] st_data;    // per lane: rd (store data)

    // ---- register-file write port (S3): instruction results, else a finished global load
    reg             s3_alu_wr, s3_smem_wr, s3_is_alu;
    reg [WB-1:0]    s3_warp;
    reg [3:0]       s3_rd;
    reg [W-1:0]     s3_wmask;
    reg [W*32-1:0]  s3_data;
    wire [W*32-1:0] smem_rdata;

    wire            s3_writes   = s3_alu_wr || s3_smem_wr;
    wire            lsu_wb_take = lsu_wb_valid && (!lsu_wb_load || !s3_writes);
    wire            rf_lsu_we   = lsu_wb_take && lsu_wb_load && lsu_wb_rd != 4'd0;
    wire [RFA-1:0]  rf_wa       = s3_writes ? {s3_warp, s3_rd} : {lsu_wb_warp, lsu_wb_rd};
    wire [W*32-1:0] rf_wd       = s3_smem_wr ? smem_rdata :
                                  s3_alu_wr  ? (s3_is_alu ? alu_res : s3_data) : lsu_wb_data;
    wire [W-1:0]    rf_wmask    = s3_writes ? s3_wmask : (rf_lsu_we ? lsu_wb_mask : {W{1'b0}});

    wire [4:0] sr_sel = s2_imm18[4:0];

    // special registers that are the same for every lane of the warp
    reg [31:0] sr_shared;
    always @* begin
        case (sr_sel)
            `RB_SR_NTID:   sr_shared = ntid;
            `RB_SR_CTAID:  sr_shared = ctaid_q;
            `RB_SR_NCTAID: sr_shared = nctaid;
            `RB_SR_WARPID: sr_shared = {{(32-WB){1'b0}}, s2_warp};
            `RB_SR_COREID: sr_shared = CORE_ID;
            `RB_SR_CLOCK:  sr_shared = clock_cnt;
            default:       sr_shared = (sr_sel[4:3] == 2'b01) ? args[sr_sel[2:0]*32 +: 32] : 32'd0;
        endcase
    end

    // ---- matrix unit (MMA): the sequencer is further down. It runs the
    // mma/mmaz in slot ex_slot of warp ex_warp, row ex_k of the tile per step;
    // a row's result is written back one cycle later (wb_*), from the ALUs'
    // second stage.
    reg  [W-1:0]     mmask [0:NW*MS-1];  // per slot: the tile columns (lanes) its operation writes
    reg  [NW*MS-1:0] mzero;              // per slot: the operation is mmaz
    reg              ex_busy;
    reg  [WB-1:0]    ex_warp;
    reg  [SB-1:0]    ex_slot;
    reg  [LB-1:0]    ex_k;
    reg              wb_v, wb_last;
    reg  [WB-1:0]    wb_w;
    reg  [SB-1:0]    wb_s;
    reg  [LB-1:0]    wb_k;
    wire             mma_enq  = s2_valid && (s2_mma || s2_mmaz);   // copy operands into a slot
    wire [SB-1:0]    enq_slot = mq_next[s2_warp] + mq_cnt[s2_warp];  // the first free one (MS: power of two)
    wire [WB+SB-1:0] enq_at   = mslot(s2_warp, enq_slot);
    wire [WB+SB-1:0] ex_at    = mslot(ex_warp, ex_slot);
    wire [WB+SB-1:0] wb_at    = mslot(wb_w, wb_s);
    // a step waits while the instruction in S2 needs the dot-product path
    // (dp4a, dp4ab) or the tile's read port (mmar). s2_mmar alone is not
    // enough to take the port: an mmar dropped in S1 leaves it set.
    wire            mmar_rd = s2_valid && s2_mmar;
    wire            ex_go   = MMA && ex_busy && !(s2_valid && s2_dot) && !mmar_rd;
    wire [W*32-1:0] mx_all;          // per lane: rs1 from the executing warp's slot
    reg  [31:0]     x_row;           // row ex_k of A: lane ex_k's rs1, for every lane
    always @* begin : p_xrow
        integer i;
        x_row = mx_all[31:0];
        for (i = 1; i < W; i = i + 1)
            if (ex_k == i) x_row = mx_all[i*32 +: 32];
        if (mzero[ex_at]) x_row = 32'd0;     // mmaz: every product and the old value are zero
    end

    // dp4ab: one lane's rs1 operand is broadcast to every lane's ALU
    wire [W*32-1:0] qa_all;
    reg  [31:0]     qa_lane;
    always @* begin : p_bcast
        integer i;
        qa_lane = qa_all[31:0];
        for (i = 1; i < W; i = i + 1)
            if (s2_lane == i) qa_lane = qa_all[i*32 +: 32];
    end

    genvar gl;
    generate
        for (gl = 0; gl < W; gl = gl + 1) begin : g_lane
            localparam [LB-1:0] LANE = gl;
            wire [31:0] qa, qb, qc, alu_y, dot_y, acc_q, mw_q;
            assign qa_all[gl*32 +: 32] = qa;
            rb_regfile #(.AW(RFA)) u_rf (
                .clk(clk), .ra(rf_ra), .rb(rf_rb), .rc(rf_rc),
                .qa(qa), .qb(qb), .qc(qc),
                .we(rf_wmask[gl]), .wa(rf_wa), .wd(rf_wd[gl*32 +: 32])
            );
            // two stages: operands in S2, result (alu_y) in S3
            rb_alu #(.MUL_WIDTH(MUL_WIDTH), .ENABLE_DP4A(ENABLE_DP4A)) u_alu (
                .clk(clk), .fn(s2_bcast ? `RB_FN_DP4A : s2_op[3:0]), .a(qa),
                .a_dot(s2_bcast ? qa_lane : qa), .b(s2_imm ? s2_imm18 : qb), .c(qc),
                .dot_sel(ex_go), .x_mma(x_row), .w_mma(mw_q),
                .c_mma(mzero[ex_at] ? 32'd0 : acc_q), .y(alu_y), .y_dot(dot_y)
            );
            if (MMA) begin : g_mma
                reg [31:0] acc [0:NW*W-1];   // this lane's column of every warp's tile
                reg [31:0] mxs [0:NW*MS-1];  // the slots: this lane's rs1 and rs2 for each operation
                reg [31:0] mws [0:NW*MS-1];
`ifndef SYNTHESIS
                integer i;
                initial for (i = 0; i < NW * W; i = i + 1) acc[i] = 32'd0;
`endif
                always @(posedge clk) begin
                    if (mma_enq) begin
                        mxs[enq_at] <= qa;
                        mws[enq_at] <= qb;
                    end
                    if (wb_v && mmask[wb_at][gl]) acc[{wb_w, wb_k}] <= dot_y;
                end
                // one read port (LUT RAM): mmar in S2, else the unit's next step
                assign acc_q = acc[mmar_rd ? {s2_warp, s2_lane} : {ex_warp, ex_k}];
                assign mw_q  = mws[ex_at];
                assign mx_all[gl*32 +: 32] = mxs[ex_at];
            end else begin : g_no_mma
                assign acc_q = 32'd0;
                assign mw_q  = 32'd0;
                assign mx_all[gl*32 +: 32] = 32'd0;
            end
            wire [31:0] tid = {{(32-WB-LB){1'b0}}, s2_warp, LANE};
            wire [31:0] srv = (sr_sel == `RB_SR_TID)    ? tid :
                              (sr_sel == `RB_SR_LANEID) ? {{(32-LB){1'b0}}, LANE} :
                              (sr_sel == `RB_SR_GTID)   ? base_q + tid : sr_shared;
            assign alu_res[gl*32 +: 32]   = alu_y;
            assign ex_result[gl*32 +: 32] = s2_mmar ? acc_q :
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
    wire [SDB-1:0]     top_slot  = spv[SDB-1:0] - 1'b1;
    wire [WB+SDB-1:0]  top_idx   = {s2_warp, top_slot};
    wire [WB+SDB-1:0]  push_idx  = {s2_warp, spv[SDB-1:0]};   // used only when not full
    wire [IMEM_AW-1:0] top_pc    = stk_pc[top_idx];
    wire [W-1:0]       pop_mask  = stk_mask[top_idx] & ~xmask[s2_warp];

    wire [IMEM_AW-1:0] target  = s2_imm18[IMEM_AW-1:0];
    wire [IMEM_AW-1:0] pc_next = s2_pc + 1'b1;

    wire         want_push = s2_valid && (s2_ssy || divergent);
    wire         do_push   = want_push && !stk_full;
    wire [W-1:0] push_mask = s2_ssy ? M : br_taken;

    wire all_done    = &done_vec;
    wire bar_release = (&(bar_vec | done_vec)) && (|bar_vec);

    // threads that exist in a block of `ntid` threads (thermometer code)
    localparam TB = $clog2(MAXT) + 1;
    localparam [TB-1:0] MAXT_C = MAXT;
    wire [TB-1:0] ntid_c = (ntid >= MAXT) ? MAXT_C : ntid[TB-1:0];
    reg  [MAXT-1:0] thread_exists;
    always @* begin : p_exists
        integer t;
        for (t = 0; t < MAXT; t = t + 1) thread_exists[t] = t < ntid_c;
    end

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
                    amask[w]     <= thread_exists[w*W +: W];
                    xmask[w]     <= {W{1'b0}};
                    sp[w]        <= {SPW{1'b0}};
                    unwind[w]    <= 1'b0;
                    replaying[w] <= 1'b0;
                    pend[w]      <= 16'd0;
                    wstate[w]    <= thread_exists[w*W] ? WS_READY : WS_DONE;
                end
            end
        end else begin
            if (all_done && lsu_idle && (!MMA || mp_any == {NW{1'b0}}) &&
                !s1_valid && !s2_valid && !s3_valid) running <= 1'b0;

            for (w = 0; w < NW; w = w + 1) begin
                if (issue && pick == w) wstate[w] <= WS_BUSY;
                if (bar_release && wstate[w] == WS_BAR) wstate[w] <= WS_READY;
                if (!SCOREBOARD && lsu_wb_take && lsu_wb_warp == w) wstate[w] <= WS_READY;
                // scoreboard: a load the LSU accepted marks its register, its write-back clears it
                if (lsu_start && !s2_op[2] && s2_rd != 4'd0 && s2_warp == w) pend[w][s2_rd] <= 1'b1;
                if (rf_lsu_we && lsu_wb_warp == w) pend[w][lsu_wb_rd] <= 1'b0;
                // an instruction that needs a register still loading is dropped in S1; the
                // warp sleeps until those registers arrive and then fetches it again
                if (s1_hazard && s1_warp == w) begin
                    wstate[w]   <= s1_mma_wait ? WS_MMA : WS_DEP;
                    dep_need[w] <= s1_regs;
                    mma_all[w]  <= s1_is_mmar;
                end
                if (wstate[w] == WS_DEP && (pend[w] & dep_need[w]) == 16'd0) wstate[w] <= WS_READY;
                if (MMA && wstate[w] == WS_MMA && (mma_all[w] ? !mp_any[w] : mp_free[w]))
                    wstate[w] <= WS_READY;
                // parked warps retry once the LSU can take an instruction again
                if (lsu_ready && wstate[w] == WS_LSU) wstate[w] <= WS_READY;

                if (s2_valid && s2_warp == w) begin
                    unwind[w] <= 1'b0;
                    wstate[w] <= WS_READY;
                    pc[w]     <= pc_next;
                    if (is_global) begin
                        if (g_replay) begin
                            pc[w]     <= s2_pc;
                            wstate[w] <= PARK ? WS_LSU : WS_READY;   // else: spin (retry)
                        end else if (!SCOREBOARD) begin
                            wstate[w] <= WS_MEM;
                        end
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
        s3_is_alu  <= s2_alu;
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
    rb_lsu #(.W(W), .WB(WB), .GMEM_AW(GMEM_AW), .LINE(GMEM_LINE), .DEPTH(LSU_DEPTH)) u_lsu (
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

    // ======================================================= matrix unit
    // Warps queue their mma/mmaz in their own ring of MS slots; the unit serves
    // the warps round robin and each warp's operations in order, one tile row
    // per step, W steps per operation. An operation's slot is free again once
    // its last row is in the tile.
    reg  [WB-1:0] mq_pick;
    reg           mq_ok;
    always @* begin : p_mq
        integer k;
        reg [WB-1:0] cand;
        mq_pick = ex_warp;
        mq_ok   = 1'b0;
        for (k = NW; k >= 1; k = k - 1) begin   // lowest offset after ex_warp wins
            cand = ex_warp + k[WB-1:0];
            if (mq_any[cand]) begin
                mq_pick = cand;
                mq_ok   = 1'b1;
            end
        end
    end
    wire ex_last  = ex_go && ex_k == W - 1;
    wire ex_start = (!ex_busy || ex_last) && mq_ok;   // the next queued operation starts
    always @(posedge clk) begin : p_mma
        integer w;
        if (rst) begin
            for (w = 0; w < NW; w = w + 1) begin
                mq_cnt[w]  <= {MCB{1'b0}};
                mp_cnt[w]  <= {MCB{1'b0}};
                mq_next[w] <= {SB{1'b0}};
            end
            ex_busy  <= 1'b0;
            ex_warp  <= {WB{1'b0}};
            ex_slot  <= {SB{1'b0}};
            ex_k     <= {LB{1'b0}};
            wb_v     <= 1'b0;
            wb_last  <= 1'b0;
        end else begin
            wb_v    <= ex_go;
            wb_last <= ex_last;
            wb_w    <= ex_warp;
            wb_s    <= ex_slot;
            wb_k    <= ex_k;
            if (ex_go) ex_k <= ex_k + 1'b1;
            if (!ex_busy || ex_last) ex_busy <= mq_ok;
            if (ex_start) begin
                ex_warp <= mq_pick;
                ex_slot <= mq_next[mq_pick];
                ex_k    <= {LB{1'b0}};
            end
            for (w = 0; w < NW; w = w + 1) begin
                mq_cnt[w] <= mq_cnt[w] + (mma_enq && s2_warp == w) - (ex_start && mq_pick == w);
                mp_cnt[w] <= mp_cnt[w] + (mma_enq && s2_warp == w) - (wb_v && wb_last && wb_w == w);
                if (MS > 1 && ex_start && mq_pick == w) mq_next[w] <= mq_next[w] + 1'b1;
            end
            if (mma_enq) begin
                mmask[enq_at] <= M;
                mzero[enq_at] <= s2_mmaz;
            end
        end
    end

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
            if (replay || s1_hazard)                cnt[`RB_PERF_REPLAYS]     <= cnt[`RB_PERF_REPLAYS] + replay + s1_hazard;
            if (lsu_start)                          cnt[`RB_PERF_GMEM_INSTR]  <= cnt[`RB_PERF_GMEM_INSTR] + 1;
            if (ev_gm_req)                          cnt[`RB_PERF_GMEM_REQS]   <= cnt[`RB_PERF_GMEM_REQS] + 1;
            if (lsu_start)                          cnt[`RB_PERF_GMEM_LANES]  <= cnt[`RB_PERF_GMEM_LANES] + popcount(M);
            if (committed && is_shared)             cnt[`RB_PERF_SMEM_INSTR]  <= cnt[`RB_PERF_SMEM_INSTR] + 1;
            if (s2_valid && is_shared)              cnt[`RB_PERF_SMEM_CYCLES] <= cnt[`RB_PERF_SMEM_CYCLES] + 1;
            if (!lsu_idle)                          cnt[`RB_PERF_LSU_BUSY]    <= cnt[`RB_PERF_LSU_BUSY] + 1;
            if (s2_valid && divergent)              cnt[`RB_PERF_DIVERGE]     <= cnt[`RB_PERF_DIVERGE] + 1;
            if (start && !running)                  cnt[`RB_PERF_BLOCKS]      <= cnt[`RB_PERF_BLOCKS] + 1;
            if (ex_go)                              cnt[`RB_PERF_MMA_STEPS]   <= cnt[`RB_PERF_MMA_STEPS] + 1;
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

`ifndef SYNTHESIS
    initial begin
        if (MMA_SLOTS < 1 || MMA_SLOTS > 255 || (MMA_SLOTS & (MMA_SLOTS - 1)) != 0)
            $fatal(1, "rb_core: MMA_SLOTS must be a power of two below 256");
    end
`endif
endmodule

`default_nettype wire
