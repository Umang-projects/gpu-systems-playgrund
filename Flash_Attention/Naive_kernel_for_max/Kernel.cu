#include<iostream>
#include<vector>
#include<cstdio>
#include<cuda_runtime.h>
using namespace std;

__global__ void find_max_naive(int* d_data, int* d_max, int n) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int x = tid; x < n; x += stride) {
        int data = d_data[x];
        atomicMax(d_max, data);
    }
}

int main() {
    int N = 1000000;
    size_t size = N * sizeof(int);

    // --- Host Memory ---
    int* h_data = (int*)malloc(size);
    int gpu_result = INT_MIN;

    // Data Fill(Random numbers)
    for (int i = 0; i < N; i++) h_data[i] = i % 1000;
    h_data[500000] = 999999;

    // --- Device Memory ---
    int* d_data, * d_max;
    cudaMalloc(&d_data, size);
    cudaMalloc(&d_max, sizeof(int));

    // Data Copy
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    int initial_max = INT_MIN;
    cudaMemcpy(d_max, &initial_max, sizeof(int), cudaMemcpyHostToDevice);

    // --- Launch Configuration ---
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Timer Setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    printf("Launching Naive Max Kernel with %d elements...\n", N);

    // --- RUN ---
    cudaEventRecord(start);
    find_max_naive << <gridSize, blockSize >> > (d_data, d_max, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Result Copy Back
    cudaMemcpy(&gpu_result, d_max, sizeof(int), cudaMemcpyDeviceToHost);

    printf("GPU Time: %.4f ms\n", milliseconds);
    printf("Found Max: %d\n", gpu_result);

    // Cleanup
    cudaFree(d_data); cudaFree(d_max);
    free(h_data);
    return 0;
}
