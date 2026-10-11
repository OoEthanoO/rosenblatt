// Framed serial-to-host-bus bridge; protocol in docs/serial.md.
// Buffer and check an entire write before issuing any bus side effects.
// One command at a time; the host must wait for each complete response.
`default_nettype none
module rb_uart_bridge #(
    parameter CLKS_PER_BIT = 434,
    parameter MAX_WORDS = 64,                    // 1..255
    parameter RX_TIMEOUT_CYCLES = CLKS_PER_BIT * 12000,
    parameter BUS_TIMEOUT_CYCLES = 1000000
) (
    input wire clk, rst, uart_rx,
    output wire uart_tx,
    output wire host_valid, host_we,
    output wire [23:0] host_addr,
    output wire [31:0] host_wdata,
    input wire host_ready, host_rvalid,
    input wire [31:0] host_rdata
);
    localparam SYNC0 = 0, SYNC1 = 1, HEADER = 2, PAYLOAD = 3,
               CRC_LO = 4, CRC_HI = 5, CHECK = 6, BUS_REQ = 7,
               BUS_READ = 8, RESPONSE = 9;
    localparam CMD_HELLO = 0, CMD_READ = 1, CMD_WRITE = 2;
    localparam OK = 0, BAD_COMMAND = 1, BAD_LENGTH = 2,
               BAD_CRC = 3, BUS_TIMEOUT = 4, BAD_ADDRESS = 5;

    wire [7:0] rx_data;
    wire rx_valid, rx_error, tx_ready;
    reg [7:0] tx_data;
    wire tx_valid;
    rb_uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (
        .clk(clk), .rst(rst), .rx(uart_rx), .data(rx_data),
        .valid(rx_valid), .framing_error(rx_error)
    );
    rb_uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (
        .clk(clk), .rst(rst), .tx(uart_tx), .data(tx_data),
        .valid(tx_valid), .ready(tx_ready)
    );

    function [15:0] crc_byte;
        input [15:0] crc;
        input [7:0] value;
        reg [15:0] c;
        integer i;
        begin
            c = crc ^ {value, 8'd0};
            for (i = 0; i < 8; i = i + 1)
                c = c[15] ? (c << 1) ^ 16'h1021 : c << 1;
            crc_byte = c;
        end
    endfunction

    reg [3:0] state;
    reg [7:0] command, sequence, count, word_index;
    reg [2:0] header_index;
    reg [1:0] byte_index;
    reg [23:0] address;
    reg [31:0] buffer [0:MAX_WORDS-1];
    reg [15:0] crc, received_crc, tx_crc;
    reg [7:0] response_status, response_count;
    reg [10:0] tx_index;
    reg [31:0] rx_age, bus_age;
    reg bus_fault;
    wire receiving = state >= SYNC1 && state <= CRC_HI;
    wire [24:0] range_end = {1'b0, address} + {17'd0, count};
    wire [10:0] payload_end = 11'd5 + {1'b0, response_count, 2'b00};
    wire [10:0] payload_offset = tx_index - 11'd5;

    assign host_valid = state == BUS_REQ && !rst;
    assign host_we = command == CMD_WRITE;
    assign host_addr = address + word_index;
    assign host_wdata = buffer[word_index];
    assign tx_valid = state == RESPONSE;
    always @* begin
        case (tx_index)
            0: tx_data = 8'h52;
            1: tx_data = 8'h42;
            2: tx_data = sequence;
            3: tx_data = response_status;
            4: tx_data = response_count;
            default: begin
                if (tx_index < payload_end)
                    tx_data = buffer[payload_offset[9:2]][payload_offset[1:0]*8 +: 8];
                else if (tx_index == payload_end) tx_data = tx_crc[7:0];
                else tx_data = tx_crc[15:8];
            end
        endcase
    end

    // Each accepted read fills one response word, including zero-latency buses.
    task finish_read;
        begin
            buffer[word_index] <= host_rdata;
            bus_age <= 0;
            if (word_index == count - 1) begin
                response_count <= count;
                state <= RESPONSE;
            end else begin
                word_index <= word_index + 1'b1;
                state <= BUS_REQ;
            end
        end
    endtask

    task bus_timeout;
        begin
            bus_fault <= 1;
            response_status <= BUS_TIMEOUT;
            response_count <= 0;
            state <= RESPONSE;
        end
    endtask

    always @(posedge clk) begin
        if (rst) begin
            state <= SYNC0;
            command <= 0;
            sequence <= 0;
            count <= 0;
            word_index <= 0;
            header_index <= 0;
            byte_index <= 0;
            address <= 0;
            crc <= 16'hffff;
            received_crc <= 0;
            tx_crc <= 16'hffff;
            response_status <= OK;
            response_count <= 0;
            tx_index <= 0;
            rx_age <= 0;
            bus_age <= 0;
            bus_fault <= 0;
        end else begin
            if (state != RESPONSE) begin
                tx_index <= 0;
                tx_crc <= 16'hffff;
            end
            if (!receiving || rx_valid) rx_age <= 0;
            else rx_age <= rx_age + 1'b1;

            // Incomplete frames and UART framing errors have no bus effects.
            // Discard silently: their sequence/header may itself be incomplete.
            if (receiving && (rx_error || (!rx_valid && rx_age == RX_TIMEOUT_CYCLES - 1)))
                state <= SYNC0;
            else case (state)
                SYNC0: if (rx_valid && rx_data == 8'h52) state <= SYNC1;
                SYNC1: if (rx_valid) begin
                    if (rx_data == 8'h42) begin
                        state <= HEADER;
                        header_index <= 0;
                        crc <= 16'hffff;
                    end else if (rx_data != 8'h52) state <= SYNC0;
                end
                HEADER: if (rx_valid) begin
                    crc <= crc_byte(crc, rx_data);
                    case (header_index)
                        0: command <= rx_data;
                        1: sequence <= rx_data;
                        2: address[7:0] <= rx_data;
                        3: address[15:8] <= rx_data;
                        4: address[23:16] <= rx_data;
                        5: begin
                            count <= rx_data;
                            word_index <= 0;
                            byte_index <= 0;
                            state <= (command == CMD_WRITE && rx_data != 0) ? PAYLOAD : CRC_LO;
                        end
                    endcase
                    header_index <= header_index + 1'b1;
                end
                PAYLOAD: if (rx_valid) begin
                    crc <= crc_byte(crc, rx_data);
                    // Drain oversize requests to their CRC without overrunning RAM.
                    if (word_index < MAX_WORDS)
                        buffer[word_index][byte_index*8 +: 8] <= rx_data;
                    if (byte_index == 3) begin
                        byte_index <= 0;
                        if (word_index == count - 1) state <= CRC_LO;
                        else word_index <= word_index + 1'b1;
                    end else byte_index <= byte_index + 1'b1;
                end
                CRC_LO: if (rx_valid) begin
                    received_crc[7:0] <= rx_data;
                    state <= CRC_HI;
                end
                CRC_HI: if (rx_valid) begin
                    received_crc[15:8] <= rx_data;
                    state <= CHECK;
                end
                CHECK: begin
                    state <= RESPONSE;
                    response_status <= OK;
                    response_count <= 0;
                    word_index <= 0;
                    bus_age <= 0;
                    if (crc != received_crc) response_status <= BAD_CRC;
                    else if (bus_fault) response_status <= BUS_TIMEOUT;
                    else if (command > CMD_WRITE) response_status <= BAD_COMMAND;
                    else if (command == CMD_HELLO) begin
                        if (count != 0) response_status <= BAD_LENGTH;
                        else if (address != 0) response_status <= BAD_ADDRESS;
                        else begin
                            buffer[0] <= (MAX_WORDS << 8) | 32'd1;
                            response_count <= 1;
                        end
                    end else if (count == 0 || count > MAX_WORDS) response_status <= BAD_LENGTH;
                    else if (range_end > 25'h1000000) response_status <= BAD_ADDRESS;
                    else state <= BUS_REQ;
                end
                BUS_REQ: begin
                    if (host_ready) begin
                        bus_age <= 0;
                        if (command == CMD_WRITE) begin
                            if (word_index == count - 1) state <= RESPONSE;
                            else word_index <= word_index + 1'b1;
                        end else if (host_rvalid) finish_read;
                        else state <= BUS_READ;
                    end else if (bus_age == BUS_TIMEOUT_CYCLES - 1) bus_timeout;
                    else bus_age <= bus_age + 1'b1;
                end
                BUS_READ: begin
                    if (host_rvalid) finish_read;
                    else if (bus_age == BUS_TIMEOUT_CYCLES - 1) bus_timeout;
                    else bus_age <= bus_age + 1'b1;
                end
                RESPONSE: if (tx_ready) begin
                    if (tx_index >= 2 && tx_index < payload_end)
                        tx_crc <= crc_byte(tx_crc, tx_data);
                    if (tx_index == payload_end + 1) state <= SYNC0;
                    else tx_index <= tx_index + 1'b1;
                end
                default: state <= SYNC0;
            endcase
        end
    end
`ifndef SYNTHESIS
    initial begin
        if (CLKS_PER_BIT < 4 || MAX_WORDS < 1 || MAX_WORDS > 255 ||
            RX_TIMEOUT_CYCLES < CLKS_PER_BIT * 12 || BUS_TIMEOUT_CYCLES < 1)
            $fatal(1, "invalid UART bridge parameters");
    end
`endif
endmodule
`default_nettype wire
