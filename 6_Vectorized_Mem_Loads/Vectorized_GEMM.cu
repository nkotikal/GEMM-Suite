#include <cuda_runtime.h>

constexpr int BM = 128, BN = 64, BK = 32; //decouple tile dims. BM = BK * TN, BN = BK * TN
constexpr int TM = 4, TN = 2;
//I add the parameter TN to also expand out more elements of B. TM loads 4 rows of A. TN will load 2 columns of B

__global__ void GEMM(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta) {
    int col = (blockDim.x * blockIdx.x + threadIdx.x)*TN;
    int row = (blockDim.y * blockIdx.y + threadIdx.y) * TM;
    int tid = threadIdx.y * blockDim.x + threadIdx.x; //flat per-block thread id.

    //Tile stride, number of threads in tile for decoupling BM BN BK from "tilesize" and each other
    int nt_tile = blockDim.x * blockDim.y; //blockDim.x = BN / TN, blockDim.y = BM / TM

    __shared__ __align__(16) float Atile[BM][BK], Btile[BK][BN]; 


    float acc[TM][TN] = {0.0f};
    
    //iterate tile along shared dim K
    #pragma unroll
    for (int tile = 0; tile < (K + BK - 1)/BK; tile++) {
        int tilecolA = tile * BK + threadIdx.x;
        //load in TM rows at a time into Atile. Can introduce a loop to make this dynamic to TM
        for (int i = 0; i < TM; i++) {
            Atile[threadIdx.y * TM + i][threadIdx.x] = (row + i < M && tilecolA < K) ? A[(row + i) * K + tilecolA] : 0.0f;
        }
        //load in TN columns at a time of B tile

        for (int i = tid; i < BK * BN / 4; i += nt_tile) { //BK * BN / 4 should be folded out during copmilation
            //row computes the row we load the float4 chunk into.
            //This is because BN is the number of elements in a tile, so BN/4 is the number of chunks per row.
            //tilecol converts the id in the block's BN/TN (BK before decoupling) domain to a multiple of 4 in the
            //tile's BN domain. Each tile maps to 4 floats this way. 
            //i%(chunks_per_row = BN/4) is done because i is the 1D index the thread maps to for this tile, and this needs
            //to be converted to the column of the chunk's start position. 
            int tilerow = i / (BN / 4), tilecol = (i % (BN/4)) * 4;
            int global_row = BK * tile + tilerow, global_col = blockIdx.x * BN + tilecol;
            //might cast row or i to uint64_t later to handle large matrices. 
            float4 vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f); //default
            if (global_row < K) {
                if (global_col + 3 < N) {
                    vec = *reinterpret_cast<const float4*>(&B[global_row * N + global_col]);
                } else {
                    //allows for N % 4 != 0, but sans vectorization benefit.
                    if (global_col < N) {
                        vec.x = B[global_row * N + global_col];
                        if (global_col + 1 < N) {
                            vec.y = B[global_row * N + global_col + 1]; 
                            if (global_col + 2 < N) {
                                vec.z = B[global_row * N + global_col + 2];
                            }
                        }
                    }
                    //and vec.w remains 0.
                }
            }
            *reinterpret_cast<float4*>(&Btile[tilerow][tilecol]) = vec;
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
    //replace BK with this. In a model where BK * TN = BN, this is the same.
    //but now, BM and TM are decoupled from BK.
    dim3 threads(BM / TM, BN / TN); 
    dim3 blocks((N + BN - 1) / (BN), 
                (M + BM - 1) / (BM));
    GEMM<<<blocks, threads>>>(A, B, C, M, N, K, alpha, beta);
    return cudaGetLastError();
}

