// top.v: the Rosenblatt GPU on a ULX3S board (Lattice ECP5).
//
// The PC talks to the GPU through the board's FTDI USB-serial chip using the
// framed protocol in docs/serial.md:
//
//     python -m ml.demo --port COM5 --baudrate 115200
//
//   LED 0   kernel running (rb_gpu busy)
//   LED 1   serial data arriving from the PC
//   LED 2   serial data going to the PC
//   LED 3-6 a 4-bit value from the PC (the demo shows the recognized digit)
//   LED 7   heartbeat (about 1.5 Hz): the clock is running
//   FIRE1   reset the GPU (also needed after a serial protocol error)
`default_nettype none

module top #(
    parameter CLK_HZ        = 25_000_000,   // the on-board oscillator
    parameter BAUD          = 115_200,
    parameter NUM_CORES     = 2,
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
    parameter LSU_DEPTH     = 4             // global-memory instructions in flight per core
) (
    input  wire       clk_25mhz,
    input  wire [6:0] btn,         // btn[1] = FIRE1 (active high)
    input  wire       ftdi_txd,    // PC -> FPGA
    output wire       ftdi_rxd,    // FPGA -> PC
    output wire [7:0] led,
    output wire       wifi_gpio0   // keep the ESP32 from resetting the board
);
    wire clk = clk_25mhz;

    // Synchronize the physical button, then hold reset for 2^16 clocks after
    // its last sampled assertion (2.62 ms at 25 MHz). Release is synchronous.
    (* async_reg = "true" *) reg [1:0] button_sync = 2'b11;
    reg [15:0] por = 16'd0;
    reg        rst = 1'b1;
    always @(posedge clk) begin
        button_sync <= {button_sync[0], btn[1]};
        if (button_sync[1]) por <= 16'd0;
        else if (!(&por)) por <= por + 1'b1;
        rst <= !(&por) || button_sync[1];
    end

    wire       busy;
    wire [7:0] user_leds;
    rb_serial_gpu #(
        .NUM_CORES(NUM_CORES), .NUM_LANES(NUM_LANES), .NUM_WARPS(NUM_WARPS),
        .IMEM_AW(IMEM_AW), .GMEM_AW(GMEM_AW), .SMEM_AW(SMEM_AW),
        .MUL_WIDTH(MUL_WIDTH), .ENABLE_DP4A(ENABLE_DP4A), .GMEM_LATENCY(GMEM_LATENCY),
        .GMEM_LINE(GMEM_LINE), .SCOREBOARD(SCOREBOARD), .LSU_DEPTH(LSU_DEPTH),
        .CLKS_PER_BIT((CLK_HZ + BAUD / 2) / BAUD)
    ) u_gpu (
        .clk(clk), .rst(rst), .uart_rx(ftdi_txd), .uart_tx(ftdi_rxd), .busy(busy),
        .user_leds(user_leds)
    );

    // ---- LEDs: stretch short serial pulses so they are visible
    reg [23:0] beat = 24'd0;
    reg [19:0] rx_seen = 20'd0, tx_seen = 20'd0;
    (* async_reg = "true" *) reg [1:0] rx_sync = 2'b11;
    always @(posedge clk) begin
        rx_sync <= {rx_sync[0], ftdi_txd};
        beat    <= beat + 1'b1;
        rx_seen <= !rx_sync[1] ? 20'hFFFFF : (rx_seen != 0 ? rx_seen - 1'b1 : 20'd0);
        tx_seen <= !ftdi_rxd ? 20'hFFFFF : (tx_seen != 0 ? tx_seen - 1'b1 : 20'd0);
    end
    assign led        = {beat[23], user_leds[3:0], tx_seen != 0, rx_seen != 0, busy};
    assign wifi_gpio0 = 1'b1;
endmodule

`default_nettype wire
