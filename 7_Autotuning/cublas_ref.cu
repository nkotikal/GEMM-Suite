#include "cublas_ref.cuh"
#include <cublas_v2.h>

static cublasHandle_t handle() {
    static cublasHandle_t h = nullptr;
    if (!h) cublasCreate(&h);
    return h;
}

// cuBLAS is column-major. Passing (B, A) with M and N swapped computes the
// row-major product without transposing anything. Same convention as run_gemm.cu.
static cublasStatus_t gemm(const float* A, const float* B, float* C,
                           int M, int N, int K, float alpha, float beta) {
    return cublasGemmEx(handle(), CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                        &alpha, B, CUDA_R_32F, N,
                                A, CUDA_R_32F, K,
                        &beta,  C, CUDA_R_32F, N,
                        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
}

cudaError_t cublasReference(const float* A, const float* B, float* C,
                            int M, int N, int K, float alpha, float beta) {
    if (gemm(A, B, C, M, N, K, alpha, beta) != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
    return cudaDeviceSynchronize();
}

float cublasTimeMs(const float* A, const float* B, float* C,
                   int M, int N, int K, float alpha, float beta,
                   int minIters, float targetMs, int* usedIters) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto timeLoop = [&](int n) {
        cudaEventRecord(start);
        for (int i = 0; i < n; i++) gemm(A, B, C, M, N, K, alpha, beta);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    };

    for (int i = 0; i < 3; i++) gemm(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();

    const float probe = timeLoop(3) / 3.0f;          // calibration
    int iters = minIters;
    if (probe > 0.0f) {
        const int wanted = static_cast<int>(targetMs / probe);
        if (wanted > iters) iters = wanted;
    }
    if (iters > 200000) iters = 200000;

    const float ms = timeLoop(iters) / iters;
    if (usedIters) *usedIters = iters;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}
