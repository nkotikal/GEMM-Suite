#include <cuda_runtime.h>


#define tilesize 32

//naive implementation
__global__ void GEMM(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta) {
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (!(row < M && col < N)) return;
    
    float sum = 0;
    for (int i = 0; i < K; i++) {
        sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

extern "C" cudaError_t launchGEMM(const float* A, const float* B, float* C,
                                   int M, int N, int K, float alpha, float beta) {
    dim3 threads(16, 16);
    dim3 blocks((M + threads.x - 1) / threads.x,
                (N + threads.y - 1) / threads.y);
    GEMM<<<blocks, threads>>>(A, B, C, M, N, K, alpha, beta);
    return cudaGetLastError();
}
