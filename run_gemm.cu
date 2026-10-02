#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

static void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

static void checkCublas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS call failed");
}

static void checkCu(CUresult status) {
    if (status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(status, &message);
        throw std::runtime_error(message ? message : "CUDA Driver API error");
    }
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
    bool all = false, flops = false, testcase = false;
    double peakTflops = 15.4;
    unsigned int seed = 0;
    float absoluteTolerance = 1.0e-3f;
    std::string kernel, output, python = "auto";
};

static void printHelp() {
    std::cout << "Usage: ./run_gemm (--kernel FILE|--all) [options]\n"
                 "  --kernel FILE       CUDA source with GEMM and launchGEMM\n"
                 "  --all               Run every matching .cu file under the current directory\n"
                 "  --testcase          Read testcase.txt beside the selected kernel and check against cuBLAS\n"
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
        else if (arg == "--testcase") options.testcase = true;
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
    if (options.testcase && options.all)
        throw std::runtime_error("--testcase requires --kernel FILE, not --all");
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

static std::string trim(std::string text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

static Options loadTestcase(Options options, const fs::path& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open testcase file: " + path.string());

    bool hasM = false, hasN = false, hasK = false;
    std::string line;
    int lineNumber = 0;
    while (std::getline(file, line)) {
        ++lineNumber;
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        const auto separator = line.find('=');
        if (separator == std::string::npos)
            throw std::runtime_error(path.string() + ":" + std::to_string(lineNumber) + ": expected key=value");
        std::string key = trim(line.substr(0, separator));
        const std::string value = trim(line.substr(separator + 1));
        std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) { return std::tolower(c); });
        if (key == "m") { options.m = std::stoi(value); hasM = true; }
        else if (key == "n") { options.n = std::stoi(value); hasN = true; }
        else if (key == "k") { options.k = std::stoi(value); hasK = true; }
        else if (key == "seed") options.seed = static_cast<unsigned int>(std::stoul(value));
        else if (key == "atol") options.absoluteTolerance = std::stof(value);
        else throw std::runtime_error(path.string() + ":" + std::to_string(lineNumber) + ": unknown key " + key);
    }
    if (!hasM || !hasN || !hasK)
        throw std::runtime_error(path.string() + ": testcase must define M, N, and K");
    if (options.m < 1 || options.n < 1 || options.k < 1 || options.absoluteTolerance < 0)
        throw std::runtime_error(path.string() + ": dimensions must be positive and atol nonnegative");
    return options;
}

using GemmLauncher = cudaError_t (*)(const float*, const float*, float*, int, int, int,
                                     float, float);

struct CompiledKernel {
    void* library = nullptr;
    GemmLauncher launch = nullptr;
    fs::path sharedObject;
    fs::path buildLog;
};

static std::string shellQuote(const std::string& value);

static std::string activeGpuArchitecture() {
    int device = 0;
    checkCuda(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    checkCuda(cudaGetDeviceProperties(&properties, device));
    const int gpuArchitecture = properties.major * 10 + properties.minor;

    FILE* compiler = popen("nvcc --list-gpu-code 2>/dev/null", "r");
    if (!compiler) throw std::runtime_error("Cannot query NVCC targets; check that nvcc is on PATH");
    int bestArchitecture = 0;
    char line[128];
    while (fgets(line, sizeof(line), compiler)) {
        std::string target = trim(line);
        if (target.compare(0, 3, "sm_") != 0) continue;
        const int architecture = std::stoi(target.substr(3));
        if (architecture <= gpuArchitecture && architecture > bestArchitecture)
            bestArchitecture = architecture;
    }
    const int compilerStatus = pclose(compiler);
    if (compilerStatus != 0 || bestArchitecture == 0)
        throw std::runtime_error("Could not find an NVCC target supported by this GPU");

    if (bestArchitecture < gpuArchitecture) {
        std::cerr << "Warning: GPU is sm_" << gpuArchitecture << ", but this NVCC supports only through sm_"
                  << bestArchitecture << "; compiling PTX for driver JIT. Use CUDA Toolkit 12.8+ for native sm_120.\n";
    }
    return "sm_" + std::to_string(bestArchitecture);
}

static CompiledKernel compileKernel(const fs::path& path, const std::string& architecture) {
    static std::atomic<unsigned int> nextId{0};
    const std::string baseName = "gemm_launcher_" + std::to_string(getpid()) + "_" +
                                 std::to_string(nextId.fetch_add(1));
    const fs::path sharedObject = fs::temp_directory_path() / (baseName + ".so");
    const fs::path buildLog = fs::temp_directory_path() / (baseName + ".log");
    const std::string command = "nvcc -O3 -std=c++17 -arch=" + architecture + " -shared -Xcompiler -fPIC " +
        shellQuote(path.string()) + " -o " + shellQuote(sharedObject.string()) +
        " > " + shellQuote(buildLog.string()) + " 2>&1";
    if (std::system(command.c_str()) != 0) {
        const std::string log = fs::exists(buildLog) ? readText(buildLog) : "No compiler log available";
        fs::remove(sharedObject);
        fs::remove(buildLog);
        throw std::runtime_error("nvcc failed for " + path.string() + ":\n" + log);
    }

    void* library = dlopen(sharedObject.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library) {
        const std::string error = dlerror();
        fs::remove(sharedObject);
        fs::remove(buildLog);
        throw std::runtime_error("Cannot load " + sharedObject.string() + ": " + error);
    }
    dlerror();
    auto launch = reinterpret_cast<GemmLauncher>(dlsym(library, "launchGEMM"));
    const char* error = dlerror();
    if (error) {
        dlclose(library);
        fs::remove(sharedObject);
        fs::remove(buildLog);
        throw std::runtime_error(path.string() + " must export launchGEMM: " + error);
    }
    return {library, launch, sharedObject, buildLog};
}

static void unloadKernel(CompiledKernel& kernel) {
    if (kernel.library) dlclose(kernel.library);
    fs::remove(kernel.sharedObject);
    fs::remove(kernel.buildLog);
    kernel.library = nullptr;
}

struct TimingStats {
    double meanMs;
    double stddevMs;
};

static TimingStats summarizeTimes(const std::vector<float>& samples) {
    double mean = 0;
    for (float sample : samples) mean += sample;
    mean /= samples.size();

    double squaredDifferences = 0;
    for (float sample : samples) {
        const double difference = sample - mean;
        squaredDifferences += difference * difference;
    }
    const double stddev = samples.size() > 1
        ? std::sqrt(squaredDifferences / (samples.size() - 1)) : 0.0;
    return {mean, stddev};
}

static TimingStats timeMs(const std::function<void()>& work, int iterations) {
    for (int i = 0; i < 3; ++i) work();
    checkCuda(cudaDeviceSynchronize());
    std::vector<cudaEvent_t> starts(iterations), ends(iterations);
    for (int i = 0; i < iterations; ++i) {
        checkCuda(cudaEventCreate(&starts[i]));
        checkCuda(cudaEventCreate(&ends[i]));
    }

    for (int i = 0; i < iterations; ++i) {
        checkCuda(cudaEventRecord(starts[i]));
        work();
        checkCuda(cudaEventRecord(ends[i]));
    }
    checkCuda(cudaEventSynchronize(ends.back()));

    std::vector<float> samples(iterations);
    for (int i = 0; i < iterations; ++i) {
        checkCuda(cudaEventElapsedTime(&samples[i], starts[i], ends[i]));
        cudaEventDestroy(starts[i]);
        cudaEventDestroy(ends[i]);
    }
    return summarizeTimes(samples);
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

static std::optional<TimingStats> pytorchTimeMs(const Options& options, const std::string& python) {
    const std::string script =
        "import sys,torch,statistics; m,n,k,iters=map(int,sys.argv[1:]); "
        "assert torch.cuda.is_available(), 'PyTorch CUDA is unavailable'; "
        "torch.manual_seed(0); a=torch.randn((m,k),device='cuda',dtype=torch.float32); "
        "b=torch.randn((k,n),device='cuda',dtype=torch.float32); "
        "fn=lambda: torch.matmul(a,b); [fn() for _ in range(3)]; "
        "starts=[torch.cuda.Event(enable_timing=True) for _ in range(iters)]; "
        "ends=[torch.cuda.Event(enable_timing=True) for _ in range(iters)]; "
        "[(starts[i].record(),fn(),ends[i].record()) for i in range(iters)]; "
        "torch.cuda.synchronize(); samples=[s.elapsed_time(e) for s,e in zip(starts,ends)]; "
        "print(sum(samples)/iters, statistics.stdev(samples) if iters > 1 else 0.0)";
    std::ostringstream command;
    command << shellQuote(python) << " -c " << shellQuote(script) << ' ' << options.m << ' ' << options.n << ' '
            << options.k << ' ' << options.iterations << " 2>/dev/null";
    FILE* pipe = popen(command.str().c_str(), "r");
    if (!pipe) return std::nullopt;
    char buffer[256];
    std::string output;
    while (fgets(buffer, sizeof(buffer), pipe)) output += buffer;
    const int status = pclose(pipe);
    if (status != 0 || output.empty()) return std::nullopt;
    TimingStats timing{};
    std::istringstream values(output);
    if (!(values >> timing.meanMs >> timing.stddevMs)) return std::nullopt;
    return timing;
}

struct DeviceMatrices {
    float* a;
    float* b;
    float* c;
    size_t outputCount;
};

struct BenchmarkResult {
    std::string name;
    TimingStats timing;
    float maxError;
};

static DeviceMatrices createDeviceMatrices(const Options& options) {
    const size_t aCount = static_cast<size_t>(options.m) * options.k;
    const size_t bCount = static_cast<size_t>(options.k) * options.n;
    const size_t cCount = static_cast<size_t>(options.m) * options.n;
    std::mt19937 random(options.seed);
    std::uniform_real_distribution<float> values(-0.5f, 0.5f);
    std::vector<float> hostA(aCount), hostB(bCount);
    for (auto& value : hostA) value = values(random);
    for (auto& value : hostB) value = values(random);

    DeviceMatrices matrices{};
    matrices.outputCount = cCount;
    checkCuda(cudaMalloc(&matrices.a, aCount * sizeof(float)));
    checkCuda(cudaMalloc(&matrices.b, bCount * sizeof(float)));
    checkCuda(cudaMalloc(&matrices.c, cCount * sizeof(float)));
    checkCuda(cudaMemcpy(matrices.a, hostA.data(), aCount * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(matrices.b, hostB.data(), bCount * sizeof(float), cudaMemcpyHostToDevice));
    return matrices;
}

static TimingStats benchmarkCublas(cublasHandle_t handle, const Options& options,
                                   const DeviceMatrices& matrices, std::vector<float>& reference) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    auto gemm = [&] {
        checkCublas(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, options.n, options.m, options.k,
            &alpha, matrices.b, CUDA_R_32F, options.n, matrices.a, CUDA_R_32F, options.k,
            &beta, matrices.c, CUDA_R_32F, options.n, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
    };

    const TimingStats timing = timeMs(gemm, options.iterations);
    reference.resize(matrices.outputCount);
    checkCuda(cudaMemcpy(reference.data(), matrices.c, matrices.outputCount * sizeof(float), cudaMemcpyDeviceToHost));
    return timing;
}

static float compareOutput(const std::vector<float>& output, const std::vector<float>& reference) {
    float maxError = 0;
    for (size_t i = 0; i < output.size(); ++i) {
        if (!std::isfinite(output[i]) || !std::isfinite(reference[i]))
            return std::numeric_limits<float>::infinity();
        maxError = std::max(maxError, std::abs(output[i] - reference[i]));
    }
    return maxError;
}

static BenchmarkResult benchmarkKernel(const fs::path& path, const Options& options,
                                       const DeviceMatrices& matrices, const std::vector<float>& reference,
                                       const std::string& architecture) {
    CompiledKernel kernel = compileKernel(path, architecture);
    auto launch = [&] {
        checkCuda(kernel.launch(matrices.a, matrices.b, matrices.c,
                                options.m, options.n, options.k, 1.0f, 0.0f));
    };

    checkCuda(cudaMemset(matrices.c, 0, matrices.outputCount * sizeof(float)));
    const TimingStats timing = timeMs(launch, options.iterations);
    std::vector<float> output(matrices.outputCount);
    checkCuda(cudaMemcpy(output.data(), matrices.c, matrices.outputCount * sizeof(float), cudaMemcpyDeviceToHost));
    unloadKernel(kernel);
    return {path.stem().string(), timing, compareOutput(output, reference)};
}

static void printReportHeader(std::ostream& report, const Options& options) {
    report << "FP32 GEMM  M=" << options.m << " N=" << options.n << " K=" << options.k
           << "  iterations=" << options.iterations << "\n"
           << std::left << std::setw(24) << "Implementation" << std::right
           << std::setw(12) << "mean ms" << std::setw(12) << "stddev ms"
           << std::setw(16) << "max abs error";
    if (options.flops) report << std::setw(14) << "TFLOP/s" << std::setw(12) << "% peak";
    report << "\n" << std::string(options.flops ? 90 : 64, '-') << "\n";
}

static void printReportRow(std::ostream& report, const Options& options, const std::string& name,
                           const TimingStats& timing, const std::string& error) {
    const double operations = 2.0 * options.m * options.n * options.k;
    report << std::fixed << std::setprecision(4)
           << std::left << std::setw(24) << name << std::right
           << std::setw(12) << timing.meanMs << std::setw(12) << timing.stddevMs
           << std::setw(16) << error;
    if (options.flops) {
        const double tflops = operations / (timing.meanMs * 1.0e9);
        report << std::setw(14) << tflops << std::setw(12) << (100.0 * tflops / options.peakTflops);
    }
    report << "\n";
}

static void printUnavailableRow(std::ostream& report, const Options& options) {
    report << std::left << std::setw(24) << "PyTorch matmul" << std::right
           << std::setw(12) << "unavailable" << std::setw(12) << "-" << std::setw(16) << "-";
    if (options.flops) report << std::setw(14) << "-" << std::setw(12) << "-";
    report << "\n";
}

static void printReportSummary(std::ostream& report, const Options& options,
                               const std::vector<BenchmarkResult>& kernels) {
    const double operations = 2.0 * options.m * options.n * options.k;
    if (options.flops)
        report << "Operations per GEMM: " << std::fixed << std::setprecision(0) << operations
               << "; peak reference: " << std::setprecision(1) << options.peakTflops << " TFLOP/s\n";
    if (options.testcase) {
        const bool passed = std::all_of(kernels.begin(), kernels.end(), [&](const BenchmarkResult& result) {
            return result.maxError <= options.absoluteTolerance;
        });
        report << "Testcase: " << (passed ? "PASS" : "FAIL")
               << " (absolute tolerance " << std::setprecision(6) << options.absoluteTolerance << ")\n";
    }
}

static int runBenchmark(Options options) {
    std::vector<fs::path> paths;
    if (options.testcase) {
        const fs::path path = resolveKernel(options.kernel);
        options = loadTestcase(options, path.parent_path() / "testcase.txt");
        paths.push_back(path);
    } else {
        paths = options.all ? discoverKernels() : std::vector<fs::path>{resolveKernel(options.kernel)};
    }
    if (paths.empty()) throw std::runtime_error("No CUDA GEMM source files found under the current directory");

    checkCu(cuInit(0));
    checkCuda(cudaSetDevice(0));
    checkCuda(cudaFree(nullptr));
    const std::string architecture = activeGpuArchitecture();
    std::cout << "Kernel target: " << architecture << "\n";
    const DeviceMatrices matrices = createDeviceMatrices(options);

    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle));
    std::vector<float> reference;
    std::ofstream reportFile;
    if (!options.output.empty()) {
        reportFile.open(options.output, std::ios::trunc);
        if (!reportFile) throw std::runtime_error("Cannot write output file: " + options.output);
    }
    auto writeBoth = [&](auto write) {
        write(std::cout);
        std::cout.flush();
        if (reportFile) {
            write(reportFile);
            reportFile.flush();
        }
    };
    writeBoth([&](std::ostream& report) { printReportHeader(report, options); });

    const TimingStats cublasTiming = benchmarkCublas(handle, options, matrices, reference);

    std::vector<BenchmarkResult> results;
    for (const auto& path : paths) {
        results.push_back(benchmarkKernel(path, options, matrices, reference, architecture));
        const BenchmarkResult& result = results.back();
        writeBoth([&](std::ostream& report) {
            printReportRow(report, options, result.name, result.timing, std::to_string(result.maxError));
        });
    }
    writeBoth([&](std::ostream& report) {
        printReportRow(report, options, "cuBLAS", cublasTiming, "-");
    });

    const std::string python = resolvePython(options);
    const auto torchTiming = pytorchTimeMs(options, python);
    if (torchTiming) {
        writeBoth([&](std::ostream& report) {
            printReportRow(report, options, "PyTorch matmul", *torchTiming, "-");
        });
    } else {
        writeBoth([&](std::ostream& report) { printUnavailableRow(report, options); });
    }

    writeBoth([&](std::ostream& report) { printReportSummary(report, options, results); });
    if (reportFile)
        std::cout << "Report saved to: " << options.output << "\n";

    cublasDestroy(handle);
    cudaFree(matrices.a);
    cudaFree(matrices.b);
    cudaFree(matrices.c);
    if (options.testcase && std::any_of(results.begin(), results.end(), [&](const BenchmarkResult& result) {
            return result.maxError > options.absoluteTolerance;
        })) return 2;
    return 0;
}

int main(int argc, char** argv) {
    try {
        return runBenchmark(parseOptions(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}