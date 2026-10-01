# GEMM-Suite

CUDA SGEMM progressive optimization study. Compare custom kernels with cuBLAS and PyTorch.

Kernels handwritten, used AI for harness.

## Build

Run from the repository root:

```bash
nvcc -O3 -std=c++17 -arch=sm_90 run_gemm.cu -lcublas -lcuda -o run_gemm
```

## Run

```bash
./run_gemm --all --m 512 --n 512 --k 512 --flops -o results.txt
./run_gemm --kernel 1_Naive/naive_gemm.cu
./run_gemm --kernel 3_Tiled/tiled_gemm.cu --testcase
```

The first command benchmarks every discovered kernel and writes `results.txt` in
the current directory. `--flops` shows TFLOP/s and percent of the default 15.4
TFLOP/s reference; set another value with `--peak-tflops N`.

## Harness

`--all` searches recursively for `.cu` files defining `GEMM`, then builds each
with `nvcc` as a temporary shared library and calls its `launchGEMM` function.
The harness compares each custom result with cuBLAS. It warms up each backend
three times, then reports the average of 20 CUDA-event-timed runs by default;
`--iters` changes the timed run count. PyTorch is measured when CUDA-enabled
PyTorch is available.

PyTorch is optional. If `.venv` has CUDA-enabled PyTorch, the harness finds it
automatically; it also checks activated virtual and Conda environments before
falling back to `python3`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
./run_gemm --all --m 512 --n 512 --k 512
```

## Add a Kernel

Each `.cu` file must define the GEMM kernel and this host launcher. Choose the
thread and block dimensions in the launcher.

```cpp
__global__ void GEMM(const float* A, const float* B, float* C,
                     int M, int N, int K, float alpha, float beta);

extern "C" cudaError_t launchGEMM(const float* A, const float* B, float* C,
                                  int M, int N, int K, float alpha, float beta) {
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y);
    GEMM<<<blocks, threads>>>(A, B, C, M, N, K, alpha, beta);
    return cudaGetLastError();
}
```

`--testcase` requires `--kernel` and reads `testcase.txt` from that kernel's
directory. It compares the result with cuBLAS and fails if max absolute error
exceeds `atol`.

```text
M=35
N=19
K=37
seed=0
atol=0.001
```