#include<iostream>
#include<vector>
#include<stdio.h>
#include<cuda_runtime.h>
#define N 10
using namespace std;


__global__ void two_sum_kernel(int* nums, int n, int target, int* result, int* found) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        if (*found == 1) return;
        for (int j = tid + 1; j < n; j++) {
            if (nums[tid] + nums[j] == target) { 
                if (atomicExch(found, 1) == 0) {
                    result[0] = tid;
                    result[1] = j;
                }
                return;
            }
        }
    }

}

int main() {
    int n = 100000;
    int target = 1234;
    vector<int> nums(n);
    for (int i = 0; i < n; ++i)nums[i] = (i * 37 + 17) % 2000;
    nums[n / 2] = 500;
    nums[n / 2 + 10] = 734;
    vector<int> result(2, -1);
    int found = 0;
    int* d_nums = nullptr, * d_result = nullptr, * d_found = nullptr;
    cudaMalloc(&d_nums, n * sizeof(int));
    cudaMalloc(&d_result, 2 * sizeof(int));
    cudaMalloc(&d_found, sizeof(int));

    cudaMemcpy(d_nums, nums.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, result.data(), 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_found, &found, sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n+threads-1)/threads;
    blocks = min(blocks, 1024);
    cout << "Launching kernel with " << blocks << " blocks, " << threads << " threads per block\n";
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    two_sum_kernel << <blocks, threads >> > (d_nums, n, target, d_result, d_found);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)cerr << "CUDA error: " << cudaGetErrorString(err) << "\n";
    cudaMemcpy(result.data(), d_result, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
    cout << "Kernel time (ms): " << ms << "\n";
    cout << "Found = " << found << ", indices = " << result[0] << ", " << result[1] << "\n";

    cudaFree(d_nums);
    cudaFree(d_result);
    cudaFree(d_found);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;

}