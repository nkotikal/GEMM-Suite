#pragma once
#include <cuda_runtime.h>

// Row-major C = alpha*A*B + beta*C, computed by cuBLAS. Used as the correctness
// reference and as the per-invocation timing control for clock drift.

// Runs once and synchronizes. Overwrites C.
cudaError_t cublasReference(const float* A, const float* B, float* C,
                            int M, int N, int K, float alpha, float beta);

// Mean milliseconds per launch, after 3 warmups. Overwrites C.
//
// The iteration count is calibrated, not fixed: a short probe measures one
// launch, then the timed loop runs however many launches it takes to cover
// targetMs (never fewer than minIters). Small shapes otherwise measure launch
// overhead and event jitter instead of the GEMM -- at 256^3 a fixed 5
// iterations gave an 8x spread on repeated measurements of the same shape.
// Writes the count actually used to *usedIters when non-null.
float cublasTimeMs(const float* A, const float* B, float* C,
                   int M, int N, int K, float alpha, float beta,
                   int minIters, float targetMs, int* usedIters);
