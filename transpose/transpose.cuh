#pragma once
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

// Kernel 1: naive global memory transpose
__global__ void transpose_naive(float* __restrict__ out,
                                const float* __restrict__ in,
                                int M, int N);

// Kernel 2: shared memory transpose (no bank conflict)
__global__ void transpose_smem(float* __restrict__ out,
                               const float* __restrict__ in,
                               int M, int N);
