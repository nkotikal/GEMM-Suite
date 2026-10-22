#include <cuda_runtime.h>
#include <stdio.h>
#include <random>
#include <algorithm>

//nvcc profile_test.cu 2D_register_tiling.cu -arch=sm_120 -o profile

extern "C" cudaError_t launchGEMM(const float* A, const float* B, float* C,
                                   int M, int N, int K, float alpha, float beta);



int main() {
	
	int M = 4096, N = 4096, K = 4096;

	float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
	float alpha = 1, beta = 1;

	cudaError_t err;
	err = cudaMalloc(&d_A, sizeof(float) * M * K);
	if (err != cudaSuccess) return 1;

	err = cudaMalloc(&d_B, sizeof(float) * K * N);
	if (err != cudaSuccess) return 1;

	err = cudaMalloc(&d_C, sizeof(float) * M * N);
	if (err != cudaSuccess) return 1; //error thing is so tedious...

	h_A = (float*)malloc(sizeof(float) * M * K);
	h_B = (float*)malloc(sizeof(float) * K * N);
	h_C = (float*)malloc(sizeof(float) * M * N);

	std::mt19937 random(0);
	std::uniform_real_distribution<float> dist (-0.5f, 0.5f);
	
	std::generate(h_A, h_A + M*K, [&](){return dist(random);});	
	std::generate(h_B, h_B + K*N, [&](){return dist(random);});
	std::generate(h_C, h_C + M*N, [&](){return dist(random);});

	cudaMemcpy(d_A, h_A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, sizeof(float) * K * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, sizeof(float) * M * N, cudaMemcpyHostToDevice);

	err = launchGEMM(d_A, d_B, d_C, M, N, K, alpha, beta);
	if (err != cudaSuccess) {
		printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
		return 1;
	}

	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
		return 1;
	}

	cudaMemcpy(h_C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

	printf("Done\n");
	printf("C[23] = %f\n", h_C[23]);


	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(h_A);
	free(h_B);
	free(h_C);
	return 0;
}
