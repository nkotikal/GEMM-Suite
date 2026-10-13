#include <cuda_runtime.h>

#define tilesize 32
#define TM 4 //row granularity of a thread
#define TN 2
//I add the parameter TN to also expand out more elements of B. TM loads 4 rows of A. TN will load 2 columns of B

__global__ void GEMM(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta) {
    int col = (blockDim.x * blockIdx.x + threadIdx.x)*TN;
    int row = (blockDim.y * blockIdx.y + threadIdx.y) * TM;

    __shared__ float Atile[tilesize*TM][tilesize], Btile[tilesize][tilesize*TN]; 

    //the "f" sums represent the first row and the "s" sums represent the second row
    float sum1f = 0.0f, sum2f = 0.0f, sum3f = 0.0f, sum4f = 0.0f, sum1s = 0.0f, sum2s = 0.0f, sum3s = 0.0f, sum4s = 0.0f;

    //iterate tile along shared dim K
    for (int tile = 0; tile < (K + tilesize - 1)/tilesize; tile++) {
        int tilecolA = tile * tilesize + threadIdx.x;
        int tilerowB = tile * tilesize + threadIdx.y;
        //load in TM rows at a time into Atile. Can introduce a loop to make this dynamic to TM
        for (int i = 0; i < TM; i++) {
            Atile[threadIdx.y * TM + i][threadIdx.x] = (row + i < M && tilecolA < K) ? A[row + i * K + tilecolA] : 0.0f;
        }
        //load in TN columns at a time of B tile
        for (int i = 0; i < TN; i++) {
            Btile[threadIdx.y][threadIdx.x * TN + i] = (tilerowB < K && col + i < N) ? B[tilerowB * N + col + i] : 0.0f;
        }
        __syncthreads();

        for (int p = 0; p < tilesize; p++) {
            float B_reuse_f = Btile[p][threadIdx.x * TN];
            float B_reuse_s = Btile[p][threadIdx.x * TN + 1];
            sum1f += Atile[threadIdx.y * TM][p] * B_reuse_f;
            sum2f += Atile[threadIdx.y * TM + 1][p] * B_reuse_f;
            sum3f += Atile[threadIdx.y * TM + 2][p] * B_reuse_f;
            sum4f += Atile[threadIdx.y * TM + 3][p] * B_reuse_f;

            sum1s += Atile[threadIdx.y * TM][p] * B_reuse_s;
            sum2s += Atile[threadIdx.y * TM + 1][p] * B_reuse_s;
            sum3s += Atile[threadIdx.y * TM + 2][p] * B_reuse_s;
            sum4s += Atile[threadIdx.y * TM + 3][p] * B_reuse_s;

        }
        
        __syncthreads();
    }

    //yeah I know this logic is starting to get a bit silly now, when we get into autotuning I'll condense into a loop.
    //I think for my learning purposes it's good for me to map all this out. 
    if (col < N) {
        if (row < M) {
            C[row * N + col] = alpha * sum1f + beta * C[row * N + col];
            if (col + 1 < N) 
                C[row * N + col + 1] = alpha * sum1s + beta * C[row * N + col + 1];
        }
        if (row + 1 < M) {
            C[(row+1) * N + col] = alpha * sum2f + beta * C[(row+1) * N + col];
            if (col + 1 < N) 
                C[(row+1) * N + col + 1] = alpha * sum2s + beta * C[(row+1) * N + col + 1];
        }
        if (row + 2 < M) {
            C[(row+2) * N + col] = alpha * sum3f + beta * C[(row+2) * N + col];
            if (col + 1 < N) 
                C[(row+2) * N + col + 1] = alpha * sum3s + beta * C[(row+2) * N + col + 1];
        }
        if (row + 3 < M) {
            C[(row+3) * N + col] = alpha * sum4f + beta * C[(row+3) * N + col];
            if (col + 1 < N) 
                C[(row+3) * N + col + 1] = alpha * sum4s + beta * C[(row+3) * N + col + 1];
        }
    }
}

extern "C" cudaError_t launchGEMM(const float* A, const float* B, float* C,
                                   int M, int N, int K, float alpha, float beta) {
    dim3 threads(tilesize, tilesize);
    dim3 blocks((N + threads.x * TN - 1) / (threads.x * TN), 
                (M + threads.y * TM - 1) / (threads.y * TM));
    GEMM<<<blocks, threads>>>(A, B, C, M, N, K, alpha, beta);
    return cudaGetLastError();
}

