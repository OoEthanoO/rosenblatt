// Board-independent serial GPU wrapper. clk and rst are synchronous; the
// board top supplies the clock/reset circuitry and pin/voltage constraints.
// Host-bus writes to word address 0x300000 (isa.HOST_BOARD_BASE) set
// `user_leds`, so the PC can show a result on the board (the GPU ignores them).
`default_nettype none
module rb_serial_gpu #(
    parameter NUM_CORES = 2, NUM_LANES = 8, NUM_WARPS = 4,
    parameter IMEM_AW = 10, GMEM_AW = 16, SMEM_AW = 10,
    parameter STACK_DEPTH = 8, MUL_WIDTH = 32, ENABLE_DP4A = 1,
    parameter GMEM_LATENCY = 8, GMEM_INTERVAL = 1, LSU_DEPTH = 4, PARK = 0, GMEM_LINE = 4,
    parameter CLKS_PER_BIT = 434, MAX_WORDS = 64,
    parameter RX_TIMEOUT_CYCLES = CLKS_PER_BIT * 12000,
    parameter BUS_TIMEOUT_CYCLES = 1000000
) (
    input wire clk, rst, uart_rx,
    output wire uart_tx, busy,
    output reg [7:0] user_leds
);
    wire host_valid, host_ready, host_we, host_rvalid;
    wire [23:0] host_addr;
    wire [31:0] host_wdata, host_rdata;

    always @(posedge clk) begin
        if (rst) user_leds <= 8'd0;
        else if (host_valid && host_ready && host_we && host_addr == 24'h300000)
            user_leds <= host_wdata[7:0];
    end
    rb_uart_bridge #(
        .CLKS_PER_BIT(CLKS_PER_BIT), .MAX_WORDS(MAX_WORDS),
        .RX_TIMEOUT_CYCLES(RX_TIMEOUT_CYCLES), .BUS_TIMEOUT_CYCLES(BUS_TIMEOUT_CYCLES)
    ) u_bridge (
        .clk(clk), .rst(rst), .uart_rx(uart_rx), .uart_tx(uart_tx),
        .host_valid(host_valid), .host_ready(host_ready), .host_we(host_we),
        .host_addr(host_addr), .host_wdata(host_wdata),
        .host_rvalid(host_rvalid), .host_rdata(host_rdata)
    );
    rb_gpu #(
        .NUM_CORES(NUM_CORES), .NUM_LANES(NUM_LANES), .NUM_WARPS(NUM_WARPS),
        .IMEM_AW(IMEM_AW), .GMEM_AW(GMEM_AW), .SMEM_AW(SMEM_AW),
        .STACK_DEPTH(STACK_DEPTH), .MUL_WIDTH(MUL_WIDTH), .ENABLE_DP4A(ENABLE_DP4A),
        .GMEM_LATENCY(GMEM_LATENCY), .GMEM_INTERVAL(GMEM_INTERVAL),
        .LSU_DEPTH(LSU_DEPTH), .PARK(PARK), .GMEM_LINE(GMEM_LINE)
    ) u_gpu (
        .clk(clk), .rst(rst), .host_valid(host_valid), .host_ready(host_ready),
        .host_we(host_we), .host_addr(host_addr), .host_wdata(host_wdata),
        .host_rvalid(host_rvalid), .host_rdata(host_rdata), .busy(busy)
    );
endmodule
`default_nettype wire
