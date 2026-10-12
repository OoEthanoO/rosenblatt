# Serial protocol

The FPGA build talks to the PC over a plain UART (8 data bits, no parity, one
stop bit). `rtl/rb_uart_bridge.v` turns framed requests into transactions on
the GPU's 32-bit host bus. `sw/rosenblatt/uart.py` (`UartBackend`) is the
matching driver, so `Device(UartBackend("COM3"))` behaves exactly like the
emulator or the simulated RTL.

The bus is word addressed (24-bit word addresses). The address map is the same
as everywhere else (`isa.HOST_*_BASE`): global memory from `0x000000`,
instruction memory from `0x100000`, control registers from `0x200000`. On the
FPGA builds one more word sits outside the GPU: writing `0x300000` sets the
board's LED register (`rb_serial_gpu.user_leds`; the ULX3S shows bits 0-3 on
LEDs 3-6).

## Frames

All multi-byte fields are little-endian. The CRC is CRC-16/CCITT-FALSE
(polynomial `0x1021`, initial value `0xFFFF`, no reflection, no final XOR; the
check value of `"123456789"` is `0x29B1`). It covers everything between the
magic bytes and the CRC.

Request (PC to FPGA):

| Bytes | Field | Notes |
| --- | --- | --- |
| 2 | magic | `"RB"` (`0x52 0x42`) |
| 1 | command | 0 HELLO, 1 READ, 2 WRITE |
| 1 | sequence | echoed in the response; the driver counts 0..255 and wraps |
| 3 | address | first word address |
| 1 | count | number of 32-bit words |
| 4 x count | payload | WRITE only |
| 2 | crc | |

Response (FPGA to PC):

| Bytes | Field | Notes |
| --- | --- | --- |
| 2 | magic | `"RB"` |
| 1 | sequence | copied from the request |
| 1 | status | see below |
| 1 | count | words that follow |
| 4 x count | payload | READ data, or the HELLO capability word |
| 2 | crc | covers sequence, status, count and payload |

## Commands

| Command | Request | Successful response |
| --- | --- | --- |
| HELLO | address 0, count 0 | one word: `MAX_WORDS << 8 \| version` (version 1) |
| READ | count 1..`MAX_WORDS` | `count` words read from consecutive addresses |
| WRITE | count 1..`MAX_WORDS` plus payload | no words; sent after every bus write has been accepted |

`MAX_WORDS` is a bridge parameter (64 in the default build). The driver reads
it with HELLO and splits longer transfers into bursts.

## Status codes

| Code | Name | Meaning |
| --- | --- | --- |
| 0 | OK | |
| 1 | BAD_COMMAND | unknown command byte |
| 2 | BAD_LENGTH | count is 0 or larger than `MAX_WORDS`, or HELLO with a count |
| 3 | BAD_CRC | the request was damaged; nothing was done |
| 4 | BUS_TIMEOUT | the GPU did not answer a bus request in time; reset required |
| 5 | BAD_ADDRESS | the range runs past the 24-bit address space, or HELLO with an address |

## Rules that keep the PC and the board in step

- **Nothing happens before the CRC checks out.** A WRITE is buffered in the
  bridge in full, and its CRC verified, before the first bus write. A corrupt
  request has no side effects.
- **One request at a time.** The PC sends a request, then waits for its
  complete response.
- **Broken frames are dropped silently.** A UART framing error, or a gap of
  about 12,000 bit times in the middle of a request, returns the bridge to
  hunting for `"RB"`. No response is sent, because the sequence byte itself may
  be the part that was lost. Overlapping magic bytes (for example `"RRB"`) are
  handled.
- **Oversized writes are drained.** A WRITE longer than `MAX_WORDS` is read to
  its end, so framing is kept, and then answered with BAD_LENGTH.
- **A bus timeout is sticky.** After BUS_TIMEOUT the bridge answers every
  request with BUS_TIMEOUT until it is reset. A late reply from the GPU can
  therefore never be mistaken for the answer to a newer request.
- **The driver never retries.** If a response is lost, corrupted or late, the
  write may or may not have happened on the board. `UartBackend` marks the
  connection failed and refuses further traffic. Reset the board and reconnect.

## Speed

A full 64-word burst is 266 bytes on the wire for 256 bytes of data. At
115,200 baud (11,520 bytes per second) that is about 23 ms, or roughly
11 KB/s, so uploading the 51 KB digit-recognition model takes about 5 seconds.
After that only the 784-byte image travels per inference. The bridge works at
any rate with at least four FPGA clocks per bit (`CLKS_PER_BIT = f_clk / baud`).
Typical FTDI USB-serial chips handle up to 3 Mbaud, so a 50 MHz design can run
at 1 Mbaud (`CLKS_PER_BIT = 50`) and upload the model in well under a second.

## Testing

- `tests/test_uart.py`: the driver against a scripted peer that fragments
  reads and writes, corrupts CRCs, drops bytes and stalls. No hardware or
  simulator is needed.
- `tests/rtl/cocotb_uart.py`: the bridge in Icarus Verilog, driven bit by bit
  through its RX/TX pins against a bus model that stalls and delays. Covers
  all 256 byte values, CRC and length errors, truncated frames, bad stop bits,
  glitches, ±1% baud mismatch, and recovery after bus timeouts.
- `tests/rtl/cocotb_serial_gpu.py`: the production `UartBackend` drives the
  complete `rb_serial_gpu` (bridge plus GPU) through the UART pins. It runs
  vector addition, a small MLP and the full trained MNIST network, and
  compares every output and performance counter with the emulator.
