#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define tilesize 32

//naive implementation
__global__ void GEMM(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    half sum = 0;
    for (int i = 0; i < K; i++) {
        sum += A[row * K + i] * B[i * N + col];
    }
    if (row < M && col < N)
        C[row * N + col] = (half)alpha * sum + (half)beta * C[row * N + col];
}
