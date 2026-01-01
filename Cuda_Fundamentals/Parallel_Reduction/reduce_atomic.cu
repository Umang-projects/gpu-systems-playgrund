#include <iostream>
#include <vector>
#include <cuda_runtime.h>

using namespace std;

// GPU kernel: atomic reduction
__global__ void reduceAtomic(const float* a, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;

    // Grid-stride loop
    for (int i = idx; i < n; i += stride) {
        local_sum += a[i];
    }

    // Atomic add to global result
    atomicAdd(result, local_sum);
}

int main() {
    int N = 1e7;
    size_t bytes = N * sizeof(float);

    // -------- Host memory --------
    vector<float> h_a(N, 1.0f);
    float h_result = 0.0f;

    // -------- Device memory --------
    float *d_a, *d_result;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_result, sizeof(float));

    // Copy input to device
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch
    int threadsPerBlock = 256;
    int blocksPerGrid = 256;   // reasonable fixed grid

    reduceAtomic<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_result, N);

    // Copy result back
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Sum = " << h_result << endl;

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_result);

    return 0;
}
