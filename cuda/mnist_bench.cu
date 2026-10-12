// mnist_bench.cu: the Rosenblatt digit network on a real NVIDIA GPU.
//
// Same integer math as ml/qmlp.py and the Rosenblatt kernels, bit for bit:
//   acc1 = b1 + W1 . x       (int8 x int8 -> int32)
//   h    = clamp((acc1 + 2^(sh-1)) >> sh, 0, 127)
//   out  = b2 + W2 . h
// run four ways, from naive to NVIDIA's own library:
//   naive     one thread per (image, neuron), byte loads, scalar multiply-add
//   dp4a      same mapping, 4 int8 products per __dp4a instruction
//   tiled     shared-memory tiles + __dp4a (the idea behind kernels/dense_tiled.s)
//   cublas    cublasGemmEx int8 GEMM (tensor cores) + a small epilogue kernel
//
// Usage: mnist_bench <dir>   where <dir> holds the files written by cuda/run.py.
// Writes <dir>/logits_<method>.bin and prints a JSON line per method.

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); std::exit(1); } } while (0)
#define CB(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)s, __FILE__, __LINE__); std::exit(1); } } while (0)

struct Dims { int batch, n_in, n_hid, n_out, n_out_pad, shift; };

template <typename T>
static std::vector<T> load(const std::string& path, size_t count) {
    std::vector<T> v(count);
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f || std::fread(v.data(), sizeof(T), count, f) != count) {
        std::fprintf(stderr, "cannot read %s\n", path.c_str());
        std::exit(1);
    }
    std::fclose(f);
    return v;
}

__device__ __forceinline__ int requant(int acc, int shift) {
    int y = (acc + (1 << (shift - 1))) >> shift;
    return min(max(y, 0), 127);
}

// ---------------------------------------------------------------- naive
// y[b][j] = bias[j] + sum_k x[b][k] * w[j][k]; one thread per output
__global__ void dense_naive(const int8_t* x, const int8_t* w, const int* bias, void* y,
                            int batch, int k_in, int n_out, int ld_out, int shift) {
    int j = blockIdx.y * blockDim.x + threadIdx.x;   // gridDim.y is limited to 65535,
    int b = blockIdx.x;                               // so the batch goes along x
    if (j >= n_out || b >= batch) return;
    const int8_t* xr = x + (size_t)b * k_in;
    const int8_t* wr = w + (size_t)j * k_in;
    int acc = bias[j];
    for (int k = 0; k < k_in; ++k) acc += (int)xr[k] * (int)wr[k];
    if (shift) static_cast<int8_t*>(y)[(size_t)b * ld_out + j] = (int8_t)requant(acc, shift);
    else       static_cast<int*>(y)[(size_t)b * ld_out + j] = acc;
}

// ----------------------------------------------------------------- dp4a
__global__ void dense_dp4a(const int8_t* x, const int8_t* w, const int* bias, void* y,
                           int batch, int k_in, int n_out, int ld_out, int shift) {
    int j = blockIdx.y * blockDim.x + threadIdx.x;   // gridDim.y is limited to 65535,
    int b = blockIdx.x;                               // so the batch goes along x
    if (j >= n_out || b >= batch) return;
    const int* xr = reinterpret_cast<const int*>(x + (size_t)b * k_in);
    const int* wr = reinterpret_cast<const int*>(w + (size_t)j * k_in);
    int acc = bias[j];
    for (int k = 0; k < k_in / 4; ++k) acc = __dp4a(xr[k], wr[k], acc);
    if (shift) static_cast<int8_t*>(y)[(size_t)b * ld_out + j] = (int8_t)requant(acc, shift);
    else       static_cast<int*>(y)[(size_t)b * ld_out + j] = acc;
}

// ---------------------------------------------------------------- tiled
// A block computes a TB x TN tile of y. Each step it stages TB rows of x and
// TN rows of w (TK bytes each) in shared memory; every thread then computes
// RB x 1 outputs from shared memory with __dp4a (register blocking along b).
// Tile: TB images x TN neurons, TK input bytes per step; each thread owns RB
// images of one neuron, so a block has TN x (TB / RB) = 64 x 16 = 1024 threads.
constexpr int TB = 64, TN = 64, TK = 64, RB = 4;
__global__ void __launch_bounds__(TN * TB / RB)
dense_tiled(const int8_t* x, const int8_t* w, const int* bias, void* y,
            int batch, int k_in, int n_out, int ld_out, int shift) {
    __shared__ int xs[TB][TK / 4 + 1];   // +1 word of padding avoids bank conflicts
    __shared__ int ws[TN][TK / 4 + 1];
    int tj = threadIdx.x % TN, tb = threadIdx.x / TN;
    int j = blockIdx.x * TN + tj;
    int b0 = blockIdx.y * TB;
    int acc[RB];
    for (int r = 0; r < RB; ++r) acc[r] = (j < n_out) ? bias[j] : 0;
    for (int k0 = 0; k0 < k_in; k0 += TK) {
        // cooperative loads: TB*TK/4 + TN*TK/4 words with blockDim threads
        for (int i = threadIdx.x; i < TB * TK / 4; i += blockDim.x) {
            int row = i / (TK / 4), col = i % (TK / 4), b = b0 + row, k = k0 + col * 4;
            xs[row][col] = (b < batch && k < k_in) ? *reinterpret_cast<const int*>(x + (size_t)b * k_in + k) : 0;
        }
        for (int i = threadIdx.x; i < TN * TK / 4; i += blockDim.x) {
            int row = i / (TK / 4), col = i % (TK / 4), jj = blockIdx.x * TN + row, k = k0 + col * 4;
            ws[row][col] = (jj < n_out && k < k_in) ? *reinterpret_cast<const int*>(w + (size_t)jj * k_in + k) : 0;
        }
        __syncthreads();
        #pragma unroll
        for (int c = 0; c < TK / 4; ++c) {
            int wv = ws[tj][c];
            #pragma unroll
            for (int r = 0; r < RB; ++r) acc[r] = __dp4a(xs[tb * RB + r][c], wv, acc[r]);
        }
        __syncthreads();
    }
    if (j >= n_out) return;
    for (int r = 0; r < RB; ++r) {
        int b = b0 + tb * RB + r;
        if (b >= batch) break;
        if (shift) static_cast<int8_t*>(y)[(size_t)b * ld_out + j] = (int8_t)requant(acc[r], shift);
        else       static_cast<int*>(y)[(size_t)b * ld_out + j] = acc[r];
    }
}

// ----------------------------------------------- cuBLAS epilogues (bias, requantize)
__global__ void epilogue_hidden(const int* acc, const int* bias, int8_t* h, int batch, int n, int shift) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch * n) h[i] = (int8_t)requant(acc[i] + bias[i % n], shift);
}
__global__ void epilogue_out(const int* acc, const int* bias, int* out, int batch, int n_pad, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch * n) out[(i / n) * n + i % n] = acc[(i / n) * n_pad + i % n] + bias[i % n];
}

struct Buffers {
    int8_t *x, *w1, *w2, *w2pad, *h;
    int *b1, *b2, *out, *acc1, *acc2;
};

static void run_method(const std::string& m, const Dims& d, const Buffers& g, cublasHandle_t blas) {
    if (m == "naive" || m == "dp4a") {
        auto kern = (m == "naive") ? dense_naive : dense_dp4a;
        dim3 blk(64), g1(d.batch, (d.n_hid + 63) / 64), g2(d.batch, (d.n_out + 63) / 64);
        kern<<<g1, blk>>>(g.x, g.w1, g.b1, g.h, d.batch, d.n_in, d.n_hid, d.n_hid, d.shift);
        kern<<<g2, blk>>>(g.h, g.w2, g.b2, g.out, d.batch, d.n_hid, d.n_out, d.n_out, 0);
    } else if (m == "tiled") {
        dim3 blk(TN * TB / RB);
        dim3 g1((d.n_hid + TN - 1) / TN, (d.batch + TB - 1) / TB), g2((d.n_out + TN - 1) / TN, (d.batch + TB - 1) / TB);
        dense_tiled<<<g1, blk>>>(g.x, g.w1, g.b1, g.h, d.batch, d.n_in, d.n_hid, d.n_hid, d.shift);
        dense_tiled<<<g2, blk>>>(g.h, g.w2, g.b2, g.out, d.batch, d.n_hid, d.n_out, d.n_out, 0);
    } else {  // cublas: C (n x batch, column-major) = W (n x k) . X^T (k x batch)
        const int one = 1, zero = 0;
        CB(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, d.n_hid, d.batch, d.n_in, &one,
                        g.w1, CUDA_R_8I, d.n_in, g.x, CUDA_R_8I, d.n_in, &zero,
                        g.acc1, CUDA_R_32I, d.n_hid, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
        int n1 = d.batch * d.n_hid;
        epilogue_hidden<<<(n1 + 255) / 256, 256>>>(g.acc1, g.b1, g.h, d.batch, d.n_hid, d.shift);
        CB(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, d.n_out_pad, d.batch, d.n_hid, &one,
                        g.w2pad, CUDA_R_8I, d.n_hid, g.h, CUDA_R_8I, d.n_hid, &zero,
                        g.acc2, CUDA_R_32I, d.n_out_pad, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
        int n2 = d.batch * d.n_out;
        epilogue_out<<<(n2 + 255) / 256, 256>>>(g.acc2, g.b2, g.out, d.batch, d.n_out_pad, d.n_out);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: mnist_bench <dir>\n"); return 2; }
    std::string dir = argv[1];
    auto hdr = load<int32_t>(dir + "/dims.bin", 6);
    Dims d{hdr[0], hdr[1], hdr[2], hdr[3], hdr[4], hdr[5]};
    auto x = load<int8_t>(dir + "/x.bin", (size_t)d.batch * d.n_in);
    auto w1 = load<int8_t>(dir + "/w1.bin", (size_t)d.n_hid * d.n_in);
    auto b1 = load<int32_t>(dir + "/b1.bin", d.n_hid);
    auto w2 = load<int8_t>(dir + "/w2.bin", (size_t)d.n_out * d.n_hid);
    auto b2 = load<int32_t>(dir + "/b2.bin", d.n_out);
    std::vector<int8_t> w2pad((size_t)d.n_out_pad * d.n_hid, 0);
    std::copy(w2.begin(), w2.end(), w2pad.begin());

    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));
    Buffers g{};
    CK(cudaMalloc(&g.x, x.size()));
    CK(cudaMalloc(&g.w1, w1.size()));
    CK(cudaMalloc(&g.w2, w2.size()));
    CK(cudaMalloc(&g.w2pad, w2pad.size()));
    CK(cudaMalloc(&g.h, (size_t)d.batch * d.n_hid));
    CK(cudaMalloc(&g.b1, b1.size() * 4));
    CK(cudaMalloc(&g.b2, b2.size() * 4));
    CK(cudaMalloc(&g.out, (size_t)d.batch * d.n_out * 4));
    CK(cudaMalloc(&g.acc1, (size_t)d.batch * d.n_hid * 4));
    CK(cudaMalloc(&g.acc2, (size_t)d.batch * d.n_out_pad * 4));
    CK(cudaMemcpy(g.w1, w1.data(), w1.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g.w2, w2.data(), w2.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g.w2pad, w2pad.data(), w2pad.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g.b1, b1.data(), b1.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g.b2, b2.data(), b2.size() * 4, cudaMemcpyHostToDevice));

    // host-to-device copy of the images, timed once
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    CK(cudaMemcpy(g.x, x.data(), x.size(), cudaMemcpyHostToDevice));
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float upload_ms = 0;
    CK(cudaEventElapsedTime(&upload_ms, e0, e1));

    cublasHandle_t blas;
    CB(cublasCreate(&blas));
    std::printf("{\"device\": \"%s\", \"sms\": %d, \"cc\": \"%d.%d\", \"l2_bytes\": %d, "
                "\"smem_per_sm\": %zu, \"regs_per_sm\": %d, \"max_threads_per_sm\": %d, "
                "\"mem_bus_bits\": %d, \"global_mem_bytes\": %zu, \"batch\": %d, \"upload_ms\": %.4f}\n",
                prop.name, prop.multiProcessorCount, prop.major, prop.minor, prop.l2CacheSize,
                prop.sharedMemPerMultiprocessor, prop.regsPerMultiprocessor, prop.maxThreadsPerMultiProcessor,
                prop.memoryBusWidth, prop.totalGlobalMem, d.batch, upload_ms);

    for (std::string m : {"naive", "dp4a", "tiled", "cublas"}) {
        CK(cudaMemset(g.out, 0, (size_t)d.batch * d.n_out * 4));
        for (int i = 0; i < 3; ++i) run_method(m, d, g, blas);   // warm up
        CK(cudaDeviceSynchronize());
        CK(cudaGetLastError());
        // repeat until at least 1 s of GPU time (long enough to sample the clock), take the mean
        int reps = 0;
        float total_ms = 0;
        CK(cudaEventRecord(e0));
        auto start = std::chrono::steady_clock::now();
        do {
            for (int i = 0; i < 10; ++i) run_method(m, d, g, blas);
            reps += 10;
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
        } while (std::chrono::steady_clock::now() - start < std::chrono::milliseconds(1000));
        CK(cudaEventElapsedTime(&total_ms, e0, e1));
        CK(cudaGetLastError());
        std::vector<int32_t> out((size_t)d.batch * d.n_out);
        CK(cudaMemcpy(out.data(), g.out, out.size() * 4, cudaMemcpyDeviceToHost));
        FILE* f = std::fopen((dir + "/logits_" + m + ".bin").c_str(), "wb");
        std::fwrite(out.data(), 4, out.size(), f);
        std::fclose(f);
        double per_batch_ms = total_ms / reps;
        std::printf("{\"method\": \"%s\", \"reps\": %d, \"batch_ms\": %.5f, \"ns_per_image\": %.3f}\n",
                    m.c_str(), reps, per_batch_ms, per_batch_ms * 1e6 / d.batch);
    }
    CB(cublasDestroy(blas));
    return 0;
}
