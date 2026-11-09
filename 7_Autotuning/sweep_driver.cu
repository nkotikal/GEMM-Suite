// Autotuning sweep driver. One binary per tile config: the config comes from -D
// at compile time because template parameters cannot be set at runtime.
//
//   nvcc -arch=sm_120 -DCFG_BM=128 -DCFG_BN=64 -DCFG_BK=32 -DCFG_TM=4 -DCFG_TN=4 \
//        sweep_driver.cu cublas_ref.o -lcublas -o sweep
//   ./sweep [M] [N] [K] [min_iters]
//
// `min_iters` is a floor. Both the kernel and cuBLAS are timed over however many
// launches it takes to cover TARGET_MS, so small shapes get thousands of
// iterations and large ones get min_iters. The CSV reports the counts used.
//
// The macros are named CFG_* rather than BM/BN/... because the kernel's template
// parameters use those names -- -DBM=128 would rewrite `template<int BM, ...>`
// into `template<int 128, ...>` and fail to compile.
//
// Prints one CSV line. Exits nonzero if the launch fails or the result is wrong.

#include "Autotuned_GEMM.cu"
#include "cublas_ref.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#ifndef CFG_BM
#define CFG_BM 128
#endif
#ifndef CFG_BN
#define CFG_BN 64
#endif
#ifndef CFG_BK
#define CFG_BK 32
#endif
#ifndef CFG_TM
#define CFG_TM 4
#endif
#ifndef CFG_TN
#define CFG_TN 4
#endif

// Minimum wall time each timed loop should cover, in milliseconds.
static constexpr float TARGET_MS = 100.0f;

#define STR2(x) #x
#define STR(x) STR2(x)
#define CONFIG "BM" STR(CFG_BM) "_BN" STR(CFG_BN) "_BK" STR(CFG_BK) \
               "_TM" STR(CFG_TM) "_TN" STR(CFG_TN)
#define LAUNCH tunedLaunchGEMM<CFG_BM, CFG_BN, CFG_BK, CFG_TM, CFG_TN>

int main(int argc, char** argv) {
    const int M     = argc > 1 ? atoi(argv[1]) : 4096;
    const int N     = argc > 2 ? atoi(argv[2]) : 4096;
    const int K     = argc > 3 ? atoi(argv[3]) : 4096;
    const int iters = argc > 4 ? atoi(argv[4]) : 20;
    const float alpha = 1.0f, beta = 1.0f;  // beta != 0 so the C read path is exercised

    std::vector<float> hA(size_t(M) * K), hB(size_t(K) * N), hC(size_t(M) * N);
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    auto gen = [&] { return dist(rng); };
    std::generate(hA.begin(), hA.end(), gen);
    std::generate(hB.begin(), hB.end(), gen);
    std::generate(hC.begin(), hC.end(), gen);

    float *dA, *dB, *dC;
    if (cudaMalloc(&dA, hA.size() * 4) != cudaSuccess ||
        cudaMalloc(&dB, hB.size() * 4) != cudaSuccess ||
        cudaMalloc(&dC, hC.size() * 4) != cudaSuccess) {
        fprintf(stderr, "%s: cudaMalloc failed\n", CONFIG);
        return 1;
    }
    cudaMemcpy(dA, hA.data(), hA.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice);

    auto resetC = [&] { cudaMemcpy(dC, hC.data(), hC.size() * 4, cudaMemcpyHostToDevice); };

    // Reference from cuBLAS, starting from the same C.
    resetC();
    if (cublasReference(dA, dB, dC, M, N, K, alpha, beta) != cudaSuccess) {
        fprintf(stderr, "%s: cuBLAS reference failed\n", CONFIG);
        return 1;
    }
    std::vector<float> ref(hC.size());
    cudaMemcpy(ref.data(), dC, ref.size() * 4, cudaMemcpyDeviceToHost);

    // One clean launch for correctness.
    resetC();
    cudaError_t err = LAUNCH(dA, dB, dC, M, N, K, alpha, beta);
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: launch failed: %s\n", CONFIG, cudaGetErrorString(err));
        return 2;
    }
    std::vector<float> out(hC.size());
    cudaMemcpy(out.data(), dC, out.size() * 4, cudaMemcpyDeviceToHost);

    float maxErr = 0.0f;
    for (size_t i = 0; i < out.size(); i++)
        maxErr = std::max(maxErr, std::abs(out[i] - ref[i]));

    // Timing: 3 warmups, a calibration probe, then a loop long enough to cover
    // TARGET_MS. `iters` is a floor, not the count -- at small shapes a fixed
    // handful of launches measures launch overhead and event jitter rather than
    // the GEMM, which showed up as an 8x spread on repeated 256^3 measurements.
    for (int i = 0; i < 3; i++) LAUNCH(dA, dB, dC, M, N, K, alpha, beta);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    auto timeLoop = [&](int n) {
        cudaEventRecord(start);
        for (int i = 0; i < n; i++) LAUNCH(dA, dB, dC, M, N, K, alpha, beta);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float t = 0.0f;
        cudaEventElapsedTime(&t, start, stop);
        return t;
    };

    const float probe = timeLoop(3) / 3.0f;
    int kIters = iters;
    if (probe > 0.0f) {
        const int wanted = static_cast<int>(TARGET_MS / probe);
        if (wanted > kIters) kIters = wanted;
    }
    if (kIters > 200000) kIters = 200000;
    const float ms = timeLoop(kIters) / kIters;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // cuBLAS timed in the same process so the ratio cancels clock drift.
    int cIters = 0;
    const float refMs = cublasTimeMs(dA, dB, dC, M, N, K, alpha, beta,
                                     iters, TARGET_MS, &cIters);

    const double flops = 2.0 * M * N * K;
    printf("%s,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.6f,%d,%d\n",
           CONFIG, M, N, K, ms, refMs, ms / refMs,
           flops / (ms * 1e9), maxErr, kIters, cIters);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}
