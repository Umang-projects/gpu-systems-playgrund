//%%writefile vector_add_vectorized.cu
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// Vectorized Kernel: float4 use karega
__global__ void vectorAddVectorized(const float* a, const float* b, float* c, int n) {
    // Index calculation ab 'float4' ke hisaab se hoga
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Typecasting pointers to float4 (128-bit load/store)
    float4* a4 = (float4*)a;
    float4* b4 = (float4*)b;
    float4* c4 = (float4*)c;

    // Loop for bulk processing (4 elements at a time)
    // Note: n/4 tak hi chalega
    for (int i = idx; i < n / 4; i += stride) {
        float4 valA = a4[i]; // 1 instruction mein 4 values load
        float4 valB = b4[i]; // 1 instruction mein 4 values load
        float4 res;
        
        // Manual vector addition
        res.x = valA.x + valB.x;
        res.y = valA.y + valB.y;
        res.z = valA.z + valB.z;
        res.w = valA.w + valB.w;

        c4[i] = res; // 1 instruction mein 4 values store
    }
    // Cleanup Loop: Agar N, 4 se divide nahi hota, to bache hue elements process karo.(PEELING)
    int remaining_start = (n / 4) * 4;
    int tid = idx * 4;
}

int main() {
    // N should be multiple of 4 for this demo optimization to run clean
    int N = 1 << 26; 
    size_t bytes = N * sizeof(float);

    float *h_a, *h_b, *h_c;
    cudaMallocHost(&h_a, bytes);
    cudaMallocHost(&h_b, bytes);
    cudaMallocHost(&h_c, bytes);

    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    // Blocks ab 4 guna kam chahiye kyunki 1 thread 4x kaam kar raha hai
    int totalThreadsNeeded = N / 4; 
    int blocksPerGrid = (totalThreadsNeeded + threadsPerBlock - 1) / threadsPerBlock;
    
    // Hardware limit (SMs) ke hisaab se clip kar sakte hain for grid-stride
    if (blocksPerGrid > 32 * 80) blocksPerGrid = 32 * 80;

    cout << "Launching Vectorized (float4) Kernel..." << endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    vectorAddVectorized<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Vectorized GPU Time taken: " << milliseconds << " ms" << endl;

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    cout << "Check: " << h_c[0] << " " << h_c[1] << "..." << endl;

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFreeHost(h_a); cudaFreeHost(h_b); cudaFreeHost(h_c);

    return 0;
}