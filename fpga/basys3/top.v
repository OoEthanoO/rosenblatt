// top.v: the Rosenblatt GPU on a Digilent Basys 3 board (AMD Artix-7 XC7A35T).
//
// The 35T holds one core of the default design (8 lanes x 4 warps), so every
// kernel runs as in simulation, on half the cores. The PC talks to the GPU
// through the board's USB port, which also programs the FPGA, using the
// framed protocol in docs/serial.md:
//
//     python -m ml.demo --port COM5 --baudrate 115200
//
//   LED 0     kernel running (rb_gpu busy)
//   LED 1     serial data arriving from the PC
//   LED 2     serial data going to the PC
//   LED 3-6   a 4-bit value from the PC (the demo shows the recognized digit)
//   LED 15    heartbeat (about 1.5 Hz): the clock is running
//   display   the same value as a hex digit; a dash until the first kernel has run
//   BTNC      reset the GPU (also needed after a serial protocol error)
`default_nettype none

module top #(
    parameter CLK_HZ        = 25_000_000,   // GPU clock, divided from the 100 MHz oscillator
    parameter BAUD          = 115_200,
    parameter NUM_CORES     = 1,
    parameter NUM_LANES     = 8,
    parameter NUM_WARPS     = 4,
    parameter GMEM_AW       = 14,           // 64 KiB of global memory
    parameter SMEM_AW       = 10,           // 4 KiB of shared memory per core
    parameter IMEM_AW       = 10,
    parameter MUL_WIDTH     = 32,
    parameter ENABLE_DP4A   = 1,
    parameter GMEM_LATENCY  = 8,
    parameter GMEM_LINE     = 4,            // bytes per global-memory request
    parameter SCOREBOARD    = 0,            // 1: loads stall only the instructions that need them
    parameter LSU_DEPTH     = 4,            // global-memory instructions in flight per core
    parameter ENABLE_MMA    = 0,            // 1: the matrix unit (mma, mmaz, mmar)
    parameter MMA_SLOTS     = 1             // mma/mmaz per warp in the matrix unit at once
) (
    input  wire        clk_100mhz,
    input  wire        btnC,        // centre button (active high)
    input  wire        RsRx,        // PC -> FPGA
    output wire        RsTx,        // FPGA -> PC
    output wire [15:0] led,
    output wire [6:0]  seg,         // segments a-g, active low
    output wire        dp,          // decimal point, active low
    output wire [3:0]  an           // digit enables, active low; an[0] is the rightmost
);
    // ---- clock: an MMCM makes CLK_HZ from the oscillator (VCO at 1 GHz). A
    // simulator has no MMCM, so there the oscillator is simply divided.
    wire clk, locked;
`ifdef SYNTHESIS
    wire clk_mmcm, clk_fb;
    MMCME2_BASE #(
        .CLKIN1_PERIOD(10.0), .DIVCLK_DIVIDE(1), .CLKFBOUT_MULT_F(10.0),
        .CLKOUT0_DIVIDE_F(1.0e9 / CLK_HZ)
    ) u_mmcm (
        .CLKIN1(clk_100mhz), .CLKFBIN(clk_fb), .CLKFBOUT(clk_fb), .CLKOUT0(clk_mmcm),
        .LOCKED(locked), .RST(1'b0), .PWRDWN(1'b0),
        .CLKFBOUTB(), .CLKOUT0B(), .CLKOUT1(), .CLKOUT1B(), .CLKOUT2(), .CLKOUT2B(),
        .CLKOUT3(), .CLKOUT3B(), .CLKOUT4(), .CLKOUT5(), .CLKOUT6()
    );
    BUFG u_bufg (.I(clk_mmcm), .O(clk));
`else
    localparam HALF = 50_000_000 / CLK_HZ;   // oscillator cycles per half period
    reg [7:0] div = 8'd0;
    reg       clk_div = 1'b0;
    reg [3:0] lock_count = 4'd0;
    always @(posedge clk_100mhz) begin
        div <= (div == HALF - 1) ? 8'd0 : div + 1'b1;
        if (div == HALF - 1) clk_div <= !clk_div;
    end
    always @(posedge clk_div) if (!(&lock_count)) lock_count <= lock_count + 1'b1;
    assign clk    = clk_div;
    assign locked = &lock_count;
`endif

    // Synchronize the button, then hold reset for 2^16 clocks after its last
    // sampled press, or after the MMCM locks (2.62 ms at 25 MHz).
    (* ASYNC_REG = "TRUE" *) reg [1:0] button_sync = 2'b11;
    reg [15:0] por = 16'd0;
    reg        rst = 1'b1;
    always @(posedge clk) begin
        button_sync <= {button_sync[0], btnC};
        if (button_sync[1] || !locked) por <= 16'd0;
        else if (!(&por)) por <= por + 1'b1;
        rst <= !(&por) || button_sync[1] || !locked;
    end

    wire       busy;
    wire [7:0] user_leds;
    rb_serial_gpu #(
        .NUM_CORES(NUM_CORES), .NUM_LANES(NUM_LANES), .NUM_WARPS(NUM_WARPS),
        .IMEM_AW(IMEM_AW), .GMEM_AW(GMEM_AW), .SMEM_AW(SMEM_AW),
        .MUL_WIDTH(MUL_WIDTH), .ENABLE_DP4A(ENABLE_DP4A), .GMEM_LATENCY(GMEM_LATENCY),
        .GMEM_LINE(GMEM_LINE), .SCOREBOARD(SCOREBOARD), .LSU_DEPTH(LSU_DEPTH), .ENABLE_MMA(ENABLE_MMA),
        .MMA_SLOTS(MMA_SLOTS),
        .CLKS_PER_BIT((CLK_HZ + BAUD / 2) / BAUD)
    ) u_gpu (
        .clk(clk), .rst(rst), .uart_rx(RsRx), .uart_tx(RsTx), .busy(busy),
        .user_leds(user_leds)
    );

    // ---- LEDs: stretch short serial pulses so they are visible
    reg [23:0] beat = 24'd0;
    reg [19:0] rx_seen = 20'd0, tx_seen = 20'd0;
    (* ASYNC_REG = "TRUE" *) reg [1:0] rx_sync = 2'b11;
    always @(posedge clk) begin
        rx_sync <= {rx_sync[0], RsRx};
        beat    <= beat + 1'b1;
        rx_seen <= !rx_sync[1] ? 20'hFFFFF : (rx_seen != 0 ? rx_seen - 1'b1 : 20'd0);
        tx_seen <= !RsTx ? 20'hFFFFF : (tx_seen != 0 ? tx_seen - 1'b1 : 20'd0);
    end
    assign led = {beat[23], 8'd0, user_leds[3:0], tx_seen != 0, rx_seen != 0, busy};

    // ---- display: the value from the PC on the rightmost digit, once a kernel has run
    reg ran = 1'b0, busy_q = 1'b0;
    always @(posedge clk) begin
        busy_q <= busy;
        if (rst) ran <= 1'b0;
        else if (busy_q && !busy) ran <= 1'b1;
    end
    reg [6:0] glyph;   // segments g..a, active high
    always @* begin
        case (user_leds[3:0])
            4'h0: glyph = 7'h3F;  4'h1: glyph = 7'h06;  4'h2: glyph = 7'h5B;  4'h3: glyph = 7'h4F;
            4'h4: glyph = 7'h66;  4'h5: glyph = 7'h6D;  4'h6: glyph = 7'h7D;  4'h7: glyph = 7'h07;
            4'h8: glyph = 7'h7F;  4'h9: glyph = 7'h6F;  4'hA: glyph = 7'h77;  4'hB: glyph = 7'h7C;
            4'hC: glyph = 7'h39;  4'hD: glyph = 7'h5E;  4'hE: glyph = 7'h79;  default: glyph = 7'h71;
        endcase
        if (!ran) glyph = 7'h40;   // a dash
    end
    assign seg = ~glyph;
    assign dp  = 1'b1;
    assign an  = 4'b1110;
endmodule

`default_nettype wire
