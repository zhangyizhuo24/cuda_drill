#include "transpose.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

static float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

// ─────────────────────────────────────────────
// Kernel implementations  (fill these in)
// ─────────────────────────────────────────────

// Naive: each thread reads one element and writes to transposed position.
// out[M×N] stores B = A^T where A is M×N (row-major).
__global__ void transpose_naive(float* __restrict__ out,
                                const float* __restrict__ in,
                                int M, int N)
{
    // TODO
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    out[i * N + j] = in[j * M + i];
}

// Shared-memory tiled transpose.
// Pad the shared-memory tile by 1 column to avoid bank conflicts.
__global__ void transpose_smem(float* __restrict__ out,
                               const float* __restrict__ in,
                               int M, int N)
{
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE + 1];

    int tx = threadIdx.x, ty = threadIdx.y;

    // 每个 block 负责输入矩阵中 (blockIdx.y, blockIdx.x) 位置的 32×32 tile
    int in_row = blockIdx.y * BLOCK_SIZE + ty;
    int in_col = blockIdx.x * BLOCK_SIZE + tx;

    // Step 1: 合并读 —— 同一 warp 的 tx 连续，in_col 连续
    if (in_row < M && in_col < N)
        tile[ty][tx] = in[in_row * N + in_col];

    __syncthreads();

    // Step 2: 合并写 —— 转置后 block 落在 (blockIdx.x, blockIdx.y)
    // 同一 warp 的 tx 连续，out_col 连续
    int out_row = blockIdx.x * BLOCK_SIZE + ty;
    int out_col = blockIdx.y * BLOCK_SIZE + tx;

    if (out_row < N && out_col < M)
        out[out_row * M + out_col] = tile[tx][ty];  // 读列，padding 消除 bank conflict
}

// ─────────────────────────────────────────────
// CPU reference (for correctness check)
// ─────────────────────────────────────────────

static void transpose_cpu(float* out, const float* in, int M, int N) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            out[j * M + i] = in[i * N + j];
}

// ─────────────────────────────────────────────
// Benchmark helper
// ─────────────────────────────────────────────

typedef void (*kernel_fn)(float*, const float*, int, int);

static float bench(kernel_fn fn, float* d_out, const float* d_in,
                   int M, int N, int warmup, int iters,
                   dim3 grid, dim3 block)
{
    // warmup
    for (int i = 0; i < warmup; ++i)
        fn<<<grid, block>>>(d_out, d_in, M, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
        fn<<<grid, block>>>(d_out, d_in, M, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = elapsed_ms(start, stop) / iters;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

// ─────────────────────────────────────────────
// Correctness check
// ─────────────────────────────────────────────

static int check(const float* ref, const float* got, int n, float tol) {
    int errors = 0;
    for (int i = 0; i < n; ++i) {
        if (fabsf(ref[i] - got[i]) > tol) {
            if (errors < 5)
                fprintf(stderr, "  mismatch at [%d]: ref=%.6f got=%.6f\n",
                        i, ref[i], got[i]);
            ++errors;
        }
    }
    return errors;
}

// ─────────────────────────────────────────────
// main
// ─────────────────────────────────────────────

int main(int argc, char** argv) {
    // Matrix dimensions — override via argv if desired
    int M = (argc > 1) ? atoi(argv[1]) : 4096;
    int N = (argc > 2) ? atoi(argv[2]) : 4096;
    const int WARMUP = 5, ITERS = 20;

    printf("Transpose  M=%d  N=%d\n", M, N);

    size_t bytes_in  = (size_t)M * N * sizeof(float);
    size_t bytes_out = (size_t)N * M * sizeof(float);

    // Host buffers
    float* h_in  = (float*)malloc(bytes_in);
    float* h_ref = (float*)malloc(bytes_out);
    float* h_got = (float*)malloc(bytes_out);

    // Fill with random data
    srand(42);
    for (int i = 0; i < M * N; ++i)
        h_in[i] = (float)rand() / RAND_MAX;

    // CPU reference
    transpose_cpu(h_ref, h_in, M, N);

    // Device buffers
    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in,  bytes_in));
    CHECK_CUDA(cudaMalloc(&d_out, bytes_out));
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes_in, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // ── naive ────────────────────────────────
    printf("\n[naive]\n");
    CHECK_CUDA(cudaMemset(d_out, 0, bytes_out));
    transpose_naive<<<grid, block>>>(d_out, d_in, M, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_got, d_out, bytes_out, cudaMemcpyDeviceToHost));

    int errs = check(h_ref, h_got, N * M, 1e-5f);
    printf("  correctness: %s  (%d errors)\n", errs ? "FAIL" : "PASS", errs);

    float ms_naive = bench(transpose_naive, d_out, d_in, M, N,
                           WARMUP, ITERS, grid, block);
    double gb_naive = 2.0 * bytes_in / 1e9 / (ms_naive / 1e3);
    printf("  latency: %.3f ms   bandwidth: %.1f GB/s\n", ms_naive, gb_naive);

    // ── shared-memory ────────────────────────
    printf("\n[smem]\n");
    CHECK_CUDA(cudaMemset(d_out, 0, bytes_out));
    transpose_smem<<<grid, block>>>(d_out, d_in, M, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_got, d_out, bytes_out, cudaMemcpyDeviceToHost));

    errs = check(h_ref, h_got, N * M, 1e-5f);
    printf("  correctness: %s  (%d errors)\n", errs ? "FAIL" : "PASS", errs);

    float ms_smem = bench(transpose_smem, d_out, d_in, M, N,
                          WARMUP, ITERS, grid, block);
    double gb_smem = 2.0 * bytes_in / 1e9 / (ms_smem / 1e3);
    printf("  latency: %.3f ms   bandwidth: %.1f GB/s\n", ms_smem, gb_smem);

    // ── summary ──────────────────────────────
    printf("\nSpeedup smem / naive: %.2fx\n", ms_naive / ms_smem);
    printf("(PyTorch baseline: run bench_pytorch.py)\n");

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in); free(h_ref); free(h_got);
    return 0;
}
