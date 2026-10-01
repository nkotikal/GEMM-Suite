#include <cuda_runtime.h>

#define tilesize 32

__global__ void GEMM(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ float Atile[tilesize][tilesize], Btile[tilesize][tilesize];


    float sum = 0.0f;
    //iterate tile along shared dim K
    for (int tile = 0; tile < (K + tilesize - 1)/tilesize; tile++) {
        int tilecolA = tile * tilesize + threadIdx.x;
        int tilerowB = tile * tilesize + threadIdx.y;
        Atile[threadIdx.y][threadIdx.x] = (row < M && tilecolA < K) ? A[row * K + tilecolA] : 0.0f;
        Btile[threadIdx.y][threadIdx.x] = (tilerowB < K && col < N) ? B[tilerowB * N + col] : 0.0f;
        __syncthreads();

        for (int p = 0; p < tilesize; p++)
            sum += Atile[threadIdx.y][p] * Btile[p][threadIdx.x];
        
        __syncthreads();
    }


    if (M > row && N > col)
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

extern "C" cudaError_t launchGEMM(const float* A, const float* B, float* C,
                                   int M, int N, int K, float alpha, float beta) {
    dim3 threads(tilesize, tilesize);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y);
    GEMM<<<blocks, threads>>>(A, B, C, M, N, K, alpha, beta);
    return cudaGetLastError();
}

