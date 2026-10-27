#include <cuda_runtime.h>

constexpr int BM = 128, BN = 64, BK = 32; //decouple tile dims. BM = BK * TN, BN = BK * TN
constexpr int TM = 4, TN = 2;
//I add the parameter TN to also expand out more elements of B. TM loads 4 rows of A. TN will load 2 columns of B

__global__ void GEMM(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta) {
    int col = (blockDim.x * blockIdx.x + threadIdx.x)*TN;
    int row = (blockDim.y * blockIdx.y + threadIdx.y) * TM;
    int tid = threadIdx.y * blockDim.x + threadidx.x; //flat per-block thread id.

    

    __shared__ __align__(16) float Atile[BM][BK], Btile[BK][BN]; 


    float acc[TM][TN] = {0.0f};
    
    //iterate tile along shared dim K
    #pragma unroll
    for (int tile = 0; tile < (K + BK - 1)/BK; tile++) {
        int tilecolA = tile * BK + threadIdx.x;
        int tilerowB = tile * BK + threadIdx.y;
        //load in TM rows at a time into Atile. Can introduce a loop to make this dynamic to TM
        for (int i = 0; i < TM; i++) {
            Atile[threadIdx.y * TM + i][threadIdx.x] = (row + i < M && tilecolA < K) ? A[(row + i) * K + tilecolA] : 0.0f;
        }
        //load in TN columns at a time of B tile

        for (int i = 0; i < TN; i++) {
            Btile[threadIdx.y][threadIdx.x * TN + i] = (tilerowB < K && col + i < N) ? B[tilerowB * N + col + i] : 0.0f;
        }
        __syncthreads();
        
        for (int p = 0; p < BK; p++) {
            //preload TM rows of A at column p and TN cols of B at row p in registers
            //to minimize shared memory rereads
            float regA[TM], regB[TN]; 
            for (int i = 0; i < TM; i++) regA[i] = Atile[threadIdx.y * TM + i][p]; 
            for (int j = 0; j < TN; j++) regB[j] = Btile[p][threadIdx.x * TN + j];

            #pragma unroll
            for (int i = 0; i < TM; i++) { 
                #pragma unroll
                for(int j = 0; j < TN; j++) { 
                    acc[i][j] += regA[i] * regB[j];
                }
            }
        }
        
        __syncthreads();
    }

    //replaced the if < M, N logic with min to streamline the loop and avoid control flow
    for (int i = row; i < min(row + TM, M); i++) {
        for (int j = col; j < min(col + TN, N); j++) {
            C[i * N + j] = alpha * acc[i-row][j-col] + beta * C[i * N + j];
        }
    }
}

extern "C" cudaError_t launchGEMM(const float* A, const float* B, float* C,
                                   int M, int N, int K, float alpha, float beta) {
    dim3 threads(BK, BK);
    dim3 blocks((N + BN - 1) / (BN), 
                (M + BM - 1) / (BM));
    GEMM<<<blocks, threads>>>(A, B, C, M, N, K, alpha, beta);
    return cudaGetLastError();
}

