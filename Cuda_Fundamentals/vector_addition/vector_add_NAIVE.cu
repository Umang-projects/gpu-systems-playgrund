//%%writefile vector_add_NAIVE.cu
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

__global__ void vectorAddNaive(const float* a, const float* b, float* c, int n) {
    // Global index calculate
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Boundary check
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int N = 1 << 26; // Approx 67 Million elements
    size_t bytes = N * sizeof(float);

    // Host memory (CPU)
    float *h_a, *h_b, *h_c;
    
    // Pinned memory allocate kar rahe hain taaki transfer fast ho (cudaMallocHost)
    cudaMallocHost(&h_a, bytes);
    cudaMallocHost(&h_b, bytes);
    cudaMallocHost(&h_c, bytes);

    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // Device memory (GPU)
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // CPU se GPU data bhejna
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Grid size calculation
    int threadsPerBlock = 256;
    // Jitne elements hain, utne blocks chahiye (round up karke)
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cout << "Launching Naive Kernel with " << blocksPerGrid << " blocks..." << endl;
    
    // CUDA events timing ke liye use karenge (ye accurate hota hai GPU ke liye)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    
    // Kernel Launch
    vectorAddNaive<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // Wait karo jab tak GPU finish na kare

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Naive GPU Time taken: " << milliseconds << " ms" << endl;

    // Wapas CPU par lana result check karne ke liye
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    cout << "Check: " << h_c[0] << " " << h_c[1] << "..." << endl;

    // Free memory
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFreeHost(h_a); cudaFreeHost(h_b); cudaFreeHost(h_c);

    return 0;
}