#include <cuda_runtime.h>


template<int BM, int BN, int BK, int TM, int TN>
__global__ void GEMM(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta) {
    int col = (blockDim.x * blockIdx.x + threadIdx.x)*TN;
    int row = (blockDim.y * blockIdx.y + threadIdx.y) * TM;
    int tid = threadIdx.y * blockDim.x + threadIdx.x; //flat per-block thread id.

    //Tile stride, number of threads in tile for decoupling BM BN BK from "tilesize" and each other
    constexpr int nt_tile = (BN/TN) * (BM/TM); //blockDim.x = BN / TN, blockDim.y = BM / TM

    __shared__ __align__(16) float Atile[BK][BM], Btile[BK][BN]; 


    float acc[TM][TN] = {0.0f};
    
    //iterate tile along shared dim K
    for (int tile = 0; tile < (K + BK - 1)/BK; tile++) {

        //----------------------------------------------------A LOAD----------------------------------------------------
        //load in TM rows at a time into Atile, vectorized + transposed. More info in B loading
        //nt_tile increment: nt_tile is number of cooperating threads in a block. Jump forward to next units of unused threads
        for (int i = tid; i < BK * BM / 4; i += nt_tile) {
            int tilerow = i / (BK / 4), tilecol = (i % (BK/4)) * 4; //unlike B, we're loading along shared dim. point is loading along contiguous dim.
            int global_row = blockIdx.y * BM + tilerow, global_col = tile * BK + tilecol;

            float4 vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_row < M) {
                if ((K&3) == 0 && global_col + 3 < K) {
                    vec = *reinterpret_cast<const float4*>(&A[global_row * K + global_col]);
                } else {
                    float vec_intermediates[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    for (int offset = 0; offset < 4; offset++) {
                        if (global_col + offset < K) {
                            vec_intermediates[offset] = A[global_row * K + global_col + offset];
                        }
                    }
                    vec = make_float4(vec_intermediates[0], vec_intermediates[1], vec_intermediates[2], vec_intermediates[3]);
                }
            }
            //*reinterpret_cast<float4*>(&Atile[tilerow][tilecol]) = vec;
            //instead of above, we load into Atile in a transposed pattern so we can load contiguously with float4 later.
            //this doesn't defeat the point of vectorization because instead of 
            // LDG STS LDG STS LDG STS LDG STS, we LDG.128 STS STS STS STS
            Atile[tilecol][tilerow] = vec.x;
            Atile[tilecol + 1][tilerow] = vec.y;
            Atile[tilecol + 2][tilerow] = vec.z;
            Atile[tilecol + 3][tilerow] = vec.w;
        }

        //----------------------------------------------------B LOAD----------------------------------------------------
        //load in TN columns at a time of B tile as vectorized chunks
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
                if ((N&3) == 0 && global_col + 3 < N) { //checks in bounds and aligned
                    vec = *reinterpret_cast<const float4*>(&B[global_row * N + global_col]);
                } else {
                    //IMPORTANT: keeping for correctness/robustness for now.
                    //However, once I complete the rest of suite, I'll split into separate kernels to handle this or add forced padding. 
                    //allows for N % 4 != 0, but sans vectorization benefit. Kinda slow.
                    float vec_intermediates[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    for (int offset = 0; offset < 4; offset++) {
                        if (global_col + offset < N) {
                            vec_intermediates[offset] = B[global_row * N + global_col + offset];
                        }
                    }
                    vec = make_float4(vec_intermediates[0], vec_intermediates[1], vec_intermediates[2], vec_intermediates[3]);
                    
                }
            }
            *reinterpret_cast<float4*>(&Btile[tilerow][tilecol]) = vec;
        }
        __syncthreads();
        
        for (int p = 0; p < BK; p++) {
            //preload TM rows of A at column p and TN cols of B at row p in registers
            //to minimize shared memory rereads
            float regA[TM], regB[TN]; 
            
            //------------------------SMEM A LOAD------------------------
            if constexpr ((TM&3) == 0) { //no divergence because we evaluate at compile time. This one checks TM%4.
                #pragma unroll
                for (int i = 0; i < TM; i += 4) //to allow TM = 8. Loop will be unrolled.
                    //if the cast on the array is pushing it to local memory, will split it out later. 
                    *reinterpret_cast<float4*>(regA + i) = *reinterpret_cast<float4*>(&Atile[p][threadIdx.y * TM + i]);
            } else if constexpr ((TM&1) == 0) { //checks for even TM
                #pragma unroll
                for (int i = 0; i < TM; i+= 2)
                    *reinterpret_cast<float2*>(regA + i) = *reinterpret_cast<float2*>(&Atile[p][threadIdx.y * TM + i]);
            } else { //fallback for anything else, likely won't be reached unless I do an odd TM during autotuning lol
                #pragma unroll
                for (int i = 0; i < TM; i++)
                    regA[i] = Atile[p][threadIdx.y * TM + i];
            }

            //------------------------SMEM B LOAD------------------------
            if constexpr ((TN&3) == 0) { 
                #pragma unroll
                for (int i = 0; i < TN; i += 4) 
                    *reinterpret_cast<float4*>(regB + i) = *reinterpret_cast<float4*>(&Btile[p][threadIdx.x * TN + i]);
            } else if constexpr ((TN&1) == 0) { 
                #pragma unroll
                for (int i = 0; i < TN; i+= 2)
                    *reinterpret_cast<float2*>(regB + i) = *reinterpret_cast<float2*>(&Btile[p][threadIdx.x * TN + i]);
            } else { 
                #pragma unroll
                for (int i = 0; i < TN; i++)
                    regB[i] = Btile[p][threadIdx.x * TN + i];
            }


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

    //----------------------------------------------EPILOGUE----------------------------------------------
    //new epilogue loop. Found in the SASS that acc is in local memory, not registers
    //so fixing that by allowing for proper unrolling with compile time constants, replacing
    //dependence on M.
    //I could also explore vectorizing the global stores to get some STG.128s, but
    //that doesn't seem like a great use of time at the moment. 
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int adj_row = row + i, adj_col = col + j;
            if (adj_row < M && adj_col < N) //previous indexing logic was to avoid this check, but now we need this back for unrolling.
                C[adj_row * N + adj_col] = alpha * acc[i][j] + beta * C[adj_row * N + adj_col];
        }
    }
}

//created this wrapper so that the harness can run launchGEMM without
//worrying about setting up template and both functions can be used
template<int BM, int BN, int BK, int TM, int TN>
cudaError_t tunedLaunchGEMM(const float* A, const float* B, float* C,
                                   int M, int N, int K, float alpha, float beta) {
    dim3 threads(BN / TN, BM / TM); 
    dim3 blocks((N + BN - 1) / (BN), 
                (M + BM - 1) / (BM));
    GEMM<BM, BN, BK, TM, TN><<<blocks, threads>>>(A, B, C, M, N, K, alpha, beta);
    return cudaGetLastError();
}


extern "C" cudaError_t launchGEMM(const float* A, const float* B, float* C,
                                   int M, int N, int K, float alpha, float beta) {
    //tunedLaunchGEMM<128, 64, 32, 4, 4>(A, B, C, M, N, K, alpha, beta);
    return cudaGetLastError();
}

