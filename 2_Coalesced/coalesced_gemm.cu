#include <cuda_runtime.h>

#define tilesize 32

//coalesced implementation
__global__ void GEMM(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta) {
    int col = blockDim.x * blockIdx.x + threadIdx.x; //thread X dimension aligns with warp
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= M || col >= N) return;

    float sum = 0;
    //By setting col to the x dimension, all threads in a warp access adjacent col values,
    //allowing threads accessing B to access adjacent columns across a row
    // in a single iteration of i.
    //in the naive, it didn't help to give adjacent warpthreads acccess for A because the 
    //memory they were accessing wasn't coalesced to begin with. 
    //!!different values of "row" across values in the same warp were accessing different rows
    //in the matrix - noncontiguous mem access.!!
    //now, row is the same in a given warp so we get a broadcast.
    for (int i = 0; i < K; i++) {
        sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

extern "C" cudaError_t launchGEMM(const float* A, const float* B, float* C,
                                   int M, int N, int K, float alpha, float beta) {
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y);
    GEMM<<<blocks, threads>>>(A, B, C, M, N, K, alpha, beta);
    return cudaGetLastError();
}
