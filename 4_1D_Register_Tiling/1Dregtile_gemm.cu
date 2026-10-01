#include <cuda_runtime.h>

#define tilesize 32
#define dtilesize 64

__global__ void GEMM(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta) {
    int col = dtilesize * blockIdx.x + threadIdx.x; //need to do 64 apart now
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ float Atile[tilesize][tilesize], Btile[tilesize][dtilesize]; //allocate 64 columns for B so we can compute 2 elements per thread


    float sum1 = 0.0f;
    float sum2 = 0.0f;
    //iterate tile along shared dim K
    for (int tile = 0; tile < (K + tilesize - 1)/tilesize; tile++) {
        int tilecolA = tile * tilesize + threadIdx.x;
        int tilerowB = tile * tilesize + threadIdx.y;
        Atile[threadIdx.y][threadIdx.x] = (row < M && tilecolA < K) ? A[row * K + tilecolA] : 0.0f;
        Btile[threadIdx.y][threadIdx.x] = (tilerowB < K && col < N) ? B[tilerowB * N + col] : 0.0f;
        //was considering hoisting out col + tilesize but I decided the compiler would do it anyway so probably doesnt matter
        Btile[threadIdx.y][threadIdx.x+tilesize] = (tilerowB < K && col+tilesize < N) ? B[tilerowB * N + col+tilesize] : 0.0f;
        __syncthreads();

        for (int p = 0; p < tilesize; p++) {
            sum1 += Atile[threadIdx.y][p] * Btile[p][threadIdx.x];
            sum2 += Atile[threadIdx.y][p] * Btile[p][threadIdx.x + tilesize];
        }
        
        __syncthreads();
    }


    if (M > row && N > col) {
        C[row * N + col] = alpha * sum1 + beta * C[row * N + col];
        C[row * N + col+tilesize] = alpha * sum2 + beta * C[row * N + col+tilesize];
    }
}

extern "C" cudaError_t launchGEMM(const float* A, const float* B, float* C,
                                   int M, int N, int K, float alpha, float beta) {
    dim3 threads(tilesize, tilesize);
    dim3 blocks((N + dtilesize - 1) / dtilesize, //columns are spaced by dtilesize now, so only launch appropriate block numbers. 
                (M + tilesize - 1) / threads.y);
    GEMM<<<blocks, threads>>>(A, B, C, M, N, K, alpha, beta);
    return cudaGetLastError();
}

