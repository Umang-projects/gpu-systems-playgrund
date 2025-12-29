// vector_add_pinned_simple.cu
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// Vectorized kernel using float4
__global__ void vectorAddVectorized(const float* a,const float* b,float* c,int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int vecCount = n / 4;

    float4* a4 = (float4*)a;
    float4* b4 = (float4*)b;
    float4* c4 = (float4*)c;
    for (int i = idx; i < vecCount; i += stride) {
        float4 va = a4[i];
        float4 vb = b4[i];

        float4 vr;
        vr.x = va.x + vb.x;
        vr.y = va.y + vb.y;
        vr.z = va.z + vb.z;
        vr.w = va.w + vb.w;

        c4[i] = vr;
    }

    // Handle remaining elements (if n is not multiple of 4)
    int start = vecCount * 4;
    for (int i = start + idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int N = 1 << 26;   // ~67 million elements
    size_t bytes = N * sizeof(float);

    // -------- Pinned host memory --------
    float *h_a, *h_b, *h_c;
    cudaHostAlloc(&h_a, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_b, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_c, bytes, cudaHostAllocDefault);

    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // -------- Device memory --------
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Create stream for async operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Async copy (works fast only because host memory is pinned)
    cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, h_b, bytes, cudaMemcpyHostToDevice, stream);

    int threadsPerBlock = 256;
    int totalThreads = (N + 3) / 4;  // float4 elements
    int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    cout << "Launching vectorized kernel (pinned memory)..." << endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);

    vectorAddVectorized<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_a, d_b, d_c, N
    );

    // Copy result back asynchronously
    cudaMemcpyAsync(h_c, d_c, bytes, cudaMemcpyDeviceToHost, stream);

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cout << "Time taken: " << ms << " ms" << endl;

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);

    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
