#include <cuda_runtime.h>

#define tilesize 32

//coalesced implementation
__global__ void GEMM(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta) {
    int col = blockDim.x * blockIdx.x + threadIdx.x; //thread X dimension aligns with warp
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= M || col >= N) return;

    float sum = 0;
    for (int i = 0; i < K; i++) {
        sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}
