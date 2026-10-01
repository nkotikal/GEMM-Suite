#include <cuda_runtime.h>

#include <cstdio>

static void printCudaVersion(const char* label, int version) {
    std::printf("%-24s %d.%d\n", label, version / 1000, (version % 1000) / 10);
}

int main() {
    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "cudaGetDevice: %s\n", cudaGetErrorString(status));
        return 1;
    }

    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "cudaGetDeviceProperties: %s\n", cudaGetErrorString(status));
        return 1;
    }

    int driverVersion = 0;
    int runtimeVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    std::printf("CUDA device %d: %s\n", device, properties.name);
    std::printf("%-24s %d.%d\n", "Compute capability",
                properties.major, properties.minor);
    std::printf("%-24s %d\n", "Multiprocessors", properties.multiProcessorCount);
    std::printf("%-24s %d\n", "Warp size", properties.warpSize);
    std::printf("%-24s %d\n", "Max threads per block", properties.maxThreadsPerBlock);
    std::printf("%-24s %d\n", "Max blocks per SM", properties.maxBlocksPerMultiProcessor);
    std::printf("%-24s %d\n", "Max threads per SM", properties.maxThreadsPerMultiProcessor);
    std::printf("%-24s %d\n", "Max warps per SM",
                properties.maxThreadsPerMultiProcessor / properties.warpSize);
    std::printf("%-24s %zu MiB\n", "Global memory",
                properties.totalGlobalMem / (1024 * 1024));
    std::printf("%-24s %zu KiB\n", "Shared memory per block",
                properties.sharedMemPerBlock / 1024);
    std::printf("%-24s %d 32-bit registers\n", "Max registers per block", properties.regsPerBlock);
    std::printf("%-24s %d 32-bit registers\n", "Registers per SM", properties.regsPerMultiprocessor);
    std::printf("%-24s %lld 32-bit registers\n", "Registers across all SMs",
                static_cast<long long>(properties.regsPerMultiprocessor) * properties.multiProcessorCount);
    std::printf("%-24s %d MHz\n", "Memory clock", properties.memoryClockRate / 1000);
    std::printf("%-24s %d-bit\n", "Memory bus width", properties.memoryBusWidth);
    printCudaVersion("CUDA driver version", driverVersion);
    printCudaVersion("CUDA runtime version", runtimeVersion);
    return 0;
}