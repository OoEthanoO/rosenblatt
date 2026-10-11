// UART 8N1 receiver. Synchronize RX, confirm the start bit, then sample
// in the middle of each bit. CLKS_PER_BIT must be at least four.
`default_nettype none
module rb_uart_rx #(
    parameter CLKS_PER_BIT = 434
) (
    input wire clk, rst, rx,
    output reg [7:0] data,
    output reg valid, framing_error
);
    localparam CW = (CLKS_PER_BIT > 1) ? $clog2(CLKS_PER_BIT) : 1;
    localparam IDLE = 0, START = 1, DATA = 2, STOP = 3, BREAK = 4;
    (* ASYNC_REG = "TRUE" *) reg rx_meta, rx_sync;
    reg [2:0] state, bit_index;
    reg [CW-1:0] count;
    reg [7:0] shift;

    always @(posedge clk) begin
        if (rst) begin
            rx_meta <= 1'b1;
            rx_sync <= 1'b1;
        end else begin
            rx_meta <= rx;
            rx_sync <= rx_meta;
        end
    end
    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            count <= 0;
            bit_index <= 0;
            shift <= 0;
            data <= 0;
            valid <= 0;
            framing_error <= 0;
        end else begin
            valid <= 0;
            framing_error <= 0;
            case (state)
                IDLE: if (!rx_sync) begin
                    count <= CLKS_PER_BIT / 2 - 1;
                    state <= START;
                end
                START: if (count != 0) count <= count - 1'b1;
                       else if (rx_sync) state <= IDLE; // reject a short glitch
                       else begin
                           count <= CLKS_PER_BIT - 1;
                           bit_index <= 0;
                           state <= DATA;
                       end
                DATA: if (count != 0) count <= count - 1'b1;
                      else begin
                          shift[bit_index] <= rx_sync;
                          count <= CLKS_PER_BIT - 1;
                          if (bit_index == 7) state <= STOP;
                          else bit_index <= bit_index + 1'b1;
                      end
                STOP: if (count != 0) count <= count - 1'b1;
                      else if (rx_sync) begin
                          data <= shift;
                          valid <= 1;
                          state <= IDLE;
                      end else begin
                          framing_error <= 1;
                          state <= BREAK;
                      end
                BREAK: if (rx_sync) state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule
`default_nettype wire
