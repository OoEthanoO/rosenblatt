**Build a tiny GPU from scratch, then make it run a neural network.**

That means designing a small parallel processor in Verilog, running it on a real FPGA board, writing the software that programs it, and ending with a demo where a chip you designed recognizes handwritten digits.

It beats every other option because NVIDIA's whole bet is that accelerated computing is a full-stack problem: silicon, memory, compilers, libraries, and AI models only perform well when designed together. Most ambitious teens aiming at NVIDIA learn CUDA or train models, which only touches the top layer. This project forces you through every layer, and that cross-stack judgment is exactly what a CTO is for.

It's also close to how NVIDIA's actual CTO started. Michael Kagan has held the role since May 2020, arriving through NVIDIA's acquisition of Mellanox, which he co-founded and served as CTO. Earlier, at Intel he architected the i860XP vector processor and managed the Pentium MMX design, so his career began with designing parallel processors. Also worth noticing: he didn't climb NVIDIA's ladder. He built something NVIDIA needed.

Realistically this takes a school year of part-time work. Spend the first month or two on foundations: Nand2Tetris teaches how a computer is built up from logic gates, and HDLBits has hundreds of browser-based Verilog exercises. Build a simple CPU as a warm-up.

Then design the GPU core: several threads executing the same instruction on different data in lockstep, the SIMT model at the heart of modern NVIDIA GPUs. Keep the instruction set tiny (add, multiply, load, store, branch) and test in simulation with open-source tools like Verilator plus cocotb for Python testbenches. Adam Majmudar's open-source tiny-gpu project is a great reference, but look at it after your own first attempt, not before.

Next, move it onto an entry-level FPGA board (inexpensive Lattice boards work with the open-source Yosys toolchain) and write the software: an assembler, a matrix-multiply kernel, and finally inference for a small digit-recognition network you trained in PyTorch and quantized to 8-bit integers.

Then do the part that turns a cool build into real understanding: profile it. You'll almost certainly find the bottleneck is moving data, not doing math, which is the same problem behind NVIDIA's real investments in high-bandwidth memory and NVLink. Add a small shared memory, measure the speedup, then write the same matrix multiply in CUDA on a real NVIDIA GPU and explain why theirs is so much faster. That explanation is the most valuable thing the project produces.

For a stretch goal, Tiny Tapeout can turn a scaled-down version into actual silicon. It pools many small designs onto one shared manufactured chip, pricing depends on your design's size, with a calculator on their site, and it has fabricated over 1,500 designs since 2022.

Finally, publish everything on GitHub with a detailed write-up of your design tradeoffs. A CTO's job is making big technical bets and explaining them clearly, so the write-up matters almost as much as the hardware.

No single project gets anyone into that chair; it's a decades-long path with plenty of luck involved. But the people who end up there understand the whole stack deeply, and this is the 17-year-old version of exactly that. It'll pay off in any chip, AI, or systems career even if the goal changes along the way.