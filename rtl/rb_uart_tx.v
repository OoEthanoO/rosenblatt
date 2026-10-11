// UART 8N1 transmitter. Accept a byte on valid && ready.
`default_nettype none
module rb_uart_tx #(
    parameter CLKS_PER_BIT = 434
) (
    input wire clk, rst,
    input wire [7:0] data,
    input wire valid,
    output wire ready, tx
);
    localparam CW = (CLKS_PER_BIT > 1) ? $clog2(CLKS_PER_BIT) : 1;
    reg [CW-1:0] count;
    reg [9:0] shift;
    reg [3:0] bit_index;
    reg busy;
    assign ready = !busy;
    assign tx = busy ? shift[0] : 1'b1;
    always @(posedge clk) begin
        if (rst) begin
            count <= 0;
            shift <= 10'h3ff;
            bit_index <= 0;
            busy <= 0;
        end else if (!busy) begin
            if (valid) begin
                shift <= {1'b1, data, 1'b0};
                bit_index <= 0;
                count <= CLKS_PER_BIT - 1;
                busy <= 1;
            end
        end else if (count != 0) count <= count - 1'b1;
        else begin
            count <= CLKS_PER_BIT - 1;
            shift <= {1'b1, shift[9:1]};
            if (bit_index == 9) busy <= 0;
            else bit_index <= bit_index + 1'b1;
        end
    end
endmodule
`default_nettype wire
