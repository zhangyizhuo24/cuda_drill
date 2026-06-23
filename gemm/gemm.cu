__global__ void gemm(float* C, const float* A, const float* B, int M, int N, int K) {
    // B已经转置了
    // 每个 block 负责输入矩阵中 (blockIdx.y, blockIdx.x) 位置的 32*32 tile
    int tid_x=threadIdx.x;
    int tid_y=threadIdx.y;
    int gid_x=blockIdx.x * blockDim.x + tid_x;
    int gid_y=blockIdx.y * blockDim.y + tid_y;
    __shared__ float tile_A[BLOCK_SIZE_Y][BLOCK_SIZE_X];
    __shared__ float tile_B[BLOCK_SIZE_Y][BLOCK_SIZE_X];
    float sum = 0.0f;
    for (int k = 0; k < N; k+=BLOCK_SIZE_X) {
        __syncthreads();
        int tmp_x=k+ tid_x;
        tile_A[tid_y][tid_x] = (gid_y < M && tmp_x < N) ? A[gid_y * N + tmp_x] : 0.0f;
        tile_B[tid_y][tid_x] = (gid_x < K && (k + tid_y) < N) ? B[gid_x * N + (k + tid_y)] : 0.0f;
        __syncthreads();
        for (int i = 0; i < BLOCK_SIZE_X; ++i) {
            sum+=tile_A[tid_y][i] * tile_B[i][tid_x];
        }

    }
    C[gid_y * K + gid_x] = sum;

}