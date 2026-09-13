# GEMM-Suite
GEMM Optimizations from naive GEMM to cuBLAS GEMM performance.


Early optimizations following this guide:
  https://www.rimikawrites.com/6-step-optimization-of-gemms-in-cuda/

## Optimization Roadmap

Each numbered row is one optimization. PyTorch is the correctness and performance
reference; CUDA and Triton are the custom-kernel implementation tracks.

| # | Optimization | CUDA | Triton | PyTorch |
|---:|---|---|---|---|
| 1 | Naive GEMM | Implement | Implement | Reference |
| 2 | Coalesced global-memory access | Implement | Implement | Indirect |
| 3 | Shared-memory tiling | Implement | Implement | Indirect |
| 4 | Register tiling | Implement | Implement | Indirect |
| 5 | 1-D thread coarsening | Implement | Implement | Indirect |
| 6 | 2-D thread coarsening | Implement | Implement | Indirect |
| 7 | Vectorized memory access | Implement | Optional | Indirect |
| 8 | Warp-level tiling | Implement | Partial | Indirect |
| 9 | Multi-level tiling | Implement | Implement | Indirect |
| 10 | Double buffering | Implement | Autotune/pipeline | Indirect |
| 11 | Asynchronous memory movement | Implement | Compiler-managed | Indirect |
| 12 | Tensor cores | WMMA/CUTLASS | `tl.dot` | `matmul` backend |
| 13 | Epilogue fusion/optimization | Implement | Implement | Fused operations |
| 14 | Architecture-specific tuning | Implement | Autotune | Backend-managed |
| 15 | Kernel selection/dispatch | Implement | Autotune | Built in |

Vectorized access is separated from 2-D coarsening. Double buffering is separated
from asynchronous movement because they are related but distinct techniques.

## Benchmark Harness Skeleton

The harness should run the same shapes and dtypes for every implementation.

The main CUDA executable should handle dispatch, validation, timing, and command
line arguments. Keep matrix shapes in a small benchmark data file. Launch Triton
and PyTorch implementations from Python.

### Build

Use CMake to describe the CUDA targets and Ninja to build them. They are not
alternatives: CMake generates the build files and Ninja executes the build.

For the first standalone kernel, this direct command is enough:

nvcc -O3 -lineinfo run_gemm.cu -o run_gemm

Once there are multiple kernels, tests, or Python extensions:

cmake -S . -B build -G Ninja
cmake --build build --config Release

CUDA extensions used by PyTorch may be included in the CMake project later. Triton
kernels are launched from Python and normally do not need CMake or nvcc.

### Run

./build/run_gemm --m 4096 --n 4096 --k 4096 --dtype fp16 --kernel naive
python torch/reference.py --m 4096 --n 4096 --k 4096
python triton/gemm.py --m 4096 --n 4096 --k 4096

### Harness stages

Parse arguments, select the kernel and matrix shape, allocate and initialize A, B,
and C, check correctness against PyTorch or cuBLAS, warm up the kernel, time
repeated launches with CUDA events, and report median time, TFLOP/s, and maximum
error.

### Profiling

nsys profile -o reports/gemm ./build/run_gemm --m 4096 --n 4096 --k 4096 --kernel tiled
ncu --set full -o reports/gemm ./build/run_gemm --m 4096 --n 4096 --k 4096 --kernel tiled

Use Nsight Systems for launch/runtime behavior and Nsight Compute for memory,
occupancy, instruction, and tensor-core metrics.