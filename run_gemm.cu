#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <nvrtc.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

static void checkCu(CUresult status) {
    if (status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(status, &message);
        throw std::runtime_error(message ? message : "CUDA driver error");
    }
}

static void checkCublas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS call failed");
}

static std::string readText(const fs::path& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot read " + path.string());
    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

struct Options {
    int m = 256, n = 256, k = 256, iterations = 20;
    bool all = false, flops = false;
    double peakTflops = 15.4;
    std::string kernel, output, python = "auto";
};

static void printHelp() {
    std::cout << "Usage: ./run_gemm (--kernel FILE|--all) [options]\n"
                 "  --kernel FILE       CUDA source containing __global__ GEMM(...)\n"
                 "  --all               Run every matching .cu file under the current directory\n"
                 "  --m N --n N --k N   Matrix dimensions (default: 256 each)\n"
                 "  --iters N           Timed launches (default: 20)\n"
                 "  --flops             Show TFLOP/s and percent of theoretical peak\n"
                 "  --peak-tflops N     Theoretical peak for the percent column (default: 15.4)\n"
                 "  --python PATH       Python executable with CUDA PyTorch (default: auto-detect)\n"
                 "  -o FILE             Write the report to FILE, replacing it if it exists\n";
}

static Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("Missing value after " + arg);
            return argv[i];
        };
        if (arg == "--help" || arg == "-h") {
            printHelp();
            std::exit(0);
        } else if (arg == "--all") options.all = true;
        else if (arg == "--flops") options.flops = true;
        else if (arg == "--kernel") options.kernel = value();
        else if (arg == "--m") options.m = std::stoi(value());
        else if (arg == "--n") options.n = std::stoi(value());
        else if (arg == "--k") options.k = std::stoi(value());
        else if (arg == "--iters") options.iterations = std::stoi(value());
        else if (arg == "--peak-tflops") options.peakTflops = std::stod(value());
        else if (arg == "--python") options.python = value();
        else if (arg == "-o") options.output = value();
        else throw std::runtime_error("Unknown option: " + arg);
    }
    if (options.all == !options.kernel.empty())
        throw std::runtime_error("Specify exactly one of --kernel FILE or --all");
    if (options.m < 1 || options.n < 1 || options.k < 1 || options.iterations < 1 || options.peakTflops <= 0)
        throw std::runtime_error("Dimensions, iterations, and peak TFLOP/s must be positive");
    return options;
}

static std::vector<fs::path> discoverKernels() {
    std::vector<fs::path> paths;
    for (const auto& entry : fs::recursive_directory_iterator(".")) {
        if (!entry.is_regular_file() || entry.path().extension() != ".cu" || entry.path().filename() == "run_gemm.cu") continue;
        const std::string source = readText(entry.path());
        if (source.find("__global__") != std::string::npos && source.find("GEMM(") != std::string::npos)
            paths.push_back(entry.path());
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

static fs::path resolveKernel(const std::string& requested) {
    const fs::path wanted(requested);
    if (fs::is_regular_file(wanted)) return wanted;
    for (const auto& path : discoverKernels()) {
        const std::string stem = path.stem().string();
        const std::string shortName = stem.size() > 5 && stem.substr(stem.size() - 5) == "_gemm"
            ? stem.substr(0, stem.size() - 5) : stem;
        if (requested == stem || requested == shortName || requested == path.filename().string()) return path;
    }
    throw std::runtime_error("No GEMM kernel found for: " + requested);
}

static CUfunction compileKernel(const fs::path& path, CUmodule& module) {
    const std::string source = readText(path);
    nvrtcProgram program;
    nvrtcResult result = nvrtcCreateProgram(&program, source.c_str(), path.filename().string().c_str(), 0, nullptr, nullptr);
    if (result != NVRTC_SUCCESS) throw std::runtime_error(nvrtcGetErrorString(result));
    result = nvrtcAddNameExpression(program, "GEMM");
    if (result != NVRTC_SUCCESS) throw std::runtime_error(nvrtcGetErrorString(result));
    const char* arguments[] = {"--gpu-architecture=compute_90", "--std=c++14", "--include-path=/usr/local/cuda/include"};
    result = nvrtcCompileProgram(program, 3, arguments);
    if (result != NVRTC_SUCCESS) {
        size_t logSize = 0;
        nvrtcGetProgramLogSize(program, &logSize);
        std::string log(logSize, '\0');
        if (logSize) nvrtcGetProgramLog(program, log.data());
        nvrtcDestroyProgram(&program);
        throw std::runtime_error("NVRTC compile failed for " + path.string() + ":\n" + log);
    }
    const char* loweredName = nullptr;
    result = nvrtcGetLoweredName(program, "GEMM", &loweredName);
    if (result != NVRTC_SUCCESS) throw std::runtime_error(nvrtcGetErrorString(result));
    const std::string kernelName(loweredName);
    size_t ptxSize = 0;
    nvrtcGetPTXSize(program, &ptxSize);
    std::vector<char> ptx(ptxSize);
    nvrtcGetPTX(program, ptx.data());
    nvrtcDestroyProgram(&program);
    checkCu(cuModuleLoadData(&module, ptx.data()));
    CUfunction function;
    checkCu(cuModuleGetFunction(&function, module, kernelName.c_str()));
    return function;
}

static float timeMs(const std::function<void()>& work, int iterations) {
    for (int i = 0; i < 3; ++i) work();
    checkCuda(cudaDeviceSynchronize());
    cudaEvent_t start, end;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&end));
    checkCuda(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) work();
    checkCuda(cudaEventRecord(end));
    checkCuda(cudaEventSynchronize(end));
    float elapsed = 0;
    checkCuda(cudaEventElapsedTime(&elapsed, start, end));
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return elapsed / iterations;
}

static std::string shellQuote(const std::string& value) {
    std::string quoted = "'";
    for (char c : value) quoted += c == '\'' ? "'\\''" : std::string(1, c);
    return quoted + "'";
}

static bool hasCudaTorch(const std::string& python) {
    const std::string probe = shellQuote(python) + " -c " +
        shellQuote("import torch; assert torch.cuda.is_available()") + " >/dev/null 2>&1";
    FILE* pipe = popen(probe.c_str(), "r");
    return pipe && pclose(pipe) == 0;
}

static std::string resolvePython(const Options& options) {
    if (options.python != "auto") return options.python;

    std::vector<std::string> candidates;
    auto addEnvironment = [&](const char* name) {
        const char* root = std::getenv(name);
        if (root && *root) candidates.push_back((fs::path(root) / "bin/python").string());
    };
    addEnvironment("VIRTUAL_ENV");
    addEnvironment("CONDA_PREFIX");
    candidates.push_back(".venv/bin/python");
    candidates.push_back("venv/bin/python");
    candidates.push_back("python3");

    for (const auto& candidate : candidates)
        if ((candidate == "python3" || fs::exists(candidate)) && hasCudaTorch(candidate)) return candidate;
    return "python3";
}

static std::string pytorchTimeMs(const Options& options, const std::string& python) {
    const std::string script =
        "import sys,torch; m,n,k,iters=map(int,sys.argv[1:]); "
        "assert torch.cuda.is_available(), 'PyTorch CUDA is unavailable'; "
        "torch.manual_seed(0); a=torch.randn((m,k),device='cuda',dtype=torch.float32); "
        "b=torch.randn((k,n),device='cuda',dtype=torch.float32); "
        "fn=lambda: torch.matmul(a,b); [fn() for _ in range(3)]; "
        "s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True); "
        "s.record(); [fn() for _ in range(iters)]; e.record(); torch.cuda.synchronize(); "
        "print(s.elapsed_time(e)/iters)";
    std::ostringstream command;
    command << shellQuote(python) << " -c " << shellQuote(script) << ' ' << options.m << ' ' << options.n << ' '
            << options.k << ' ' << options.iterations << " 2>/dev/null";
    FILE* pipe = popen(command.str().c_str(), "r");
    if (!pipe) return "unavailable";
    char buffer[256];
    std::string output;
    while (fgets(buffer, sizeof(buffer), pipe)) output += buffer;
    const int status = pclose(pipe);
    if (status != 0 || output.empty()) return "unavailable";
    return output;
}

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        const std::vector<fs::path> kernels = options.all ? discoverKernels() : std::vector<fs::path>{resolveKernel(options.kernel)};
        if (kernels.empty()) throw std::runtime_error("No CUDA GEMM source files found under the current directory");

        checkCuda(cudaSetDevice(0));
        checkCuda(cudaFree(nullptr));
        checkCu(cuInit(0));

        const size_t aCount = static_cast<size_t>(options.m) * options.k;
        const size_t bCount = static_cast<size_t>(options.k) * options.n;
        const size_t cCount = static_cast<size_t>(options.m) * options.n;
        std::mt19937 random(0);
        std::uniform_real_distribution<float> values(-0.5f, 0.5f);
        std::vector<float> hostA(aCount), hostB(bCount), hostC(cCount);
        for (auto& item : hostA) item = values(random);
        for (auto& item : hostB) item = values(random);

        float *deviceA, *deviceB, *deviceC;
        checkCuda(cudaMalloc(&deviceA, aCount * sizeof(float)));
        checkCuda(cudaMalloc(&deviceB, bCount * sizeof(float)));
        checkCuda(cudaMalloc(&deviceC, cCount * sizeof(float)));
        checkCuda(cudaMemcpy(deviceA, hostA.data(), aCount * sizeof(float), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(deviceB, hostB.data(), bCount * sizeof(float), cudaMemcpyHostToDevice));

        cublasHandle_t handle;
        checkCublas(cublasCreate(&handle));
        int m = options.m, n = options.n, k = options.k;
        float kernelAlpha = 1.0f, kernelBeta = 0.0f;
        auto cublasGemm = [&] {
            checkCublas(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, options.n, options.m, options.k,
                &kernelAlpha, deviceB, CUDA_R_32F, options.n, deviceA, CUDA_R_32F, options.k,
                &kernelBeta, deviceC, CUDA_R_32F, options.n, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
        };
        const float cublasMs = timeMs(cublasGemm, options.iterations);
        checkCuda(cudaMemcpy(hostC.data(), deviceC, cCount * sizeof(float), cudaMemcpyDeviceToHost));
        const std::vector<float> reference = hostC;

        struct Row { std::string name, ms, error; };
        std::vector<Row> rows;
        const unsigned int tiles = static_cast<unsigned int>((std::max(options.m, options.n) + 15) / 16);
        for (const auto& path : kernels) {
            CUmodule module;
            CUfunction function = compileKernel(path, module);
            auto launch = [&] {
                void* args[] = {&deviceA, &deviceB, &deviceC,
                    &m, &n, &k, &kernelAlpha, &kernelBeta};
                checkCu(cuLaunchKernel(function, tiles, tiles, 1, 16, 16, 1, 0, nullptr, args, nullptr));
            };
            checkCuda(cudaMemset(deviceC, 0, cCount * sizeof(float)));
            const float elapsed = timeMs(launch, options.iterations);
            checkCuda(cudaMemcpy(hostC.data(), deviceC, cCount * sizeof(float), cudaMemcpyDeviceToHost));
            float maxError = 0;
            for (size_t i = 0; i < cCount; ++i) {
                if (!std::isfinite(hostC[i]) || !std::isfinite(reference[i])) {
                    maxError = std::numeric_limits<float>::infinity();
                    break;
                }
                maxError = std::max(maxError, std::abs(hostC[i] - reference[i]));
            }
            rows.push_back({path.stem().string(), std::to_string(elapsed), std::to_string(maxError)});
            checkCu(cuModuleUnload(module));
        }
        const std::string torchMs = pytorchTimeMs(options, resolvePython(options));
        const double operations = 2.0 * options.m * options.n * options.k; //OPS CALCULATION: 2 * M * N * K. Divide by time to get flops!
        const double cublasTflops = operations / (cublasMs * 1.0e9);

        std::ostringstream report;
        report << std::fixed << std::setprecision(4)
               << "FP32 GEMM  M=" << options.m << " N=" << options.n << " K=" << options.k
               << "  iterations=" << options.iterations << "\n"
               << std::left << std::setw(24) << "Implementation" << std::right << std::setw(12) << "ms"
               << std::setw(16) << "max abs error";
        if (options.flops) report << std::setw(14) << "TFLOP/s" << std::setw(12) << "% peak";
        report << "\n" << std::string(options.flops ? 78 : 52, '-') << "\n";
        auto printRow = [&](const std::string& name, double ms, const std::string& error) {
            const double tflops = operations / (ms * 1.0e9);
            report << std::left << std::setw(24) << name << std::right << std::setw(12) << ms
                   << std::setw(16) << error;
            if (options.flops) report << std::setw(14) << tflops << std::setw(12) << (100.0 * tflops / options.peakTflops);
            report << "\n";
        };
        for (const auto& row : rows) printRow(row.name, std::stod(row.ms), row.error);
        printRow("cuBLAS", cublasMs, "-");
        if (torchMs == "unavailable") report << std::left << std::setw(24) << "PyTorch matmul" << std::right << std::setw(12) << "unavailable" << "\n";
        else printRow("PyTorch matmul", std::stod(torchMs), "-");
        if (options.flops)
            report << "Operations per GEMM: " << std::fixed << std::setprecision(0) << operations
                   << "; peak reference: " << std::setprecision(1) << options.peakTflops << " TFLOP/s\n";

        std::cout << report.str();
        if (!options.output.empty()) {
            std::ofstream file(options.output, std::ios::trunc);
            if (!file) throw std::runtime_error("Cannot write output file: " + options.output);
            file << report.str();
            std::cout << "Report saved to: " << options.output << "\n";
        }
        cublasDestroy(handle);
        cudaFree(deviceA); cudaFree(deviceB); cudaFree(deviceC);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}