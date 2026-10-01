# GEMM-Suite

CUDA GEMM experiments. Compare custom kernels with cuBLAS and PyTorch.

Kernels handwritten, used AI for harness.

## Build

Run from the repository root:

```bash
nvcc -O3 -std=c++17 -arch=sm_90 run_gemm.cu -lcublas -lnvrtc -lcuda -o run_gemm
```

## Run

```bash
./run_gemm --all --m 512 --n 512 --k 512 --flops -o results.txt
./run_gemm --kernel 1_Naive/naive_gemm.cu
```

The first command benchmarks every discovered kernel and writes `results.txt` in
the current directory. `--flops` shows TFLOP/s and percent of the default 15.4
TFLOP/s reference; set another value with `--peak-tflops N`.

PyTorch is optional. If `.venv` has CUDA-enabled PyTorch, the harness finds it
automatically; it also checks activated virtual and Conda environments before
falling back to `python3`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
./run_gemm --all --m 512 --n 512 --k 512
```

## Add a Kernel

Add a `.cu` file with this entry point. The harness launches it with 16x16
threads; check row and column bounds in the kernel.

```cpp
__global__ void GEMM(const float* A, const float* B, float* C,
                     int M, int N, int K, float alpha, float beta);
```