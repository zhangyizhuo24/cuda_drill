__global__ void block_reduce(float* out, const float* in, int N){
    extern __shared__ float sdata[];
    int tid - thresdIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N)
        sdata[tid] = (gid < N) ? in[gid] : 0.0f;
    __scynthread();
    for (int s=blockDim.x/2; s>0; s>>=1){
        if (tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __scynthread();
    }
    if (tid == 0)
        out[blockIdx.x] = sdata[0];
}