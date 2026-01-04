#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

using namespace std;

// --- CONFIGURATION ---
const int N = 50 * 1024 * 1024; // 50 Million Elements (Approx 200MB)
const float DROPOUT_PROB = 0.5f; // 50% neurons drop honge
const float SCALE = 1.0f / (1.0f - DROPOUT_PROB); // Inverted Dropout Scale

// ====================================================
// APPROACH 1: NAIVE (Separate Kernels)
// ====================================================

// Kernel A: Sirf ReLU karta hai
__global__ void relu_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        output[idx] = fmaxf(0.0f, val); // ReLU Logic: Max(0, x)
    }
}

// Kernel B: Sirf Dropout karta hai
__global__ void dropout_kernel(float* input, float* output, float* mask, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Agar mask > 0.5 hai to rakho, warna 0 kar do
        output[idx] = (mask[idx] > DROPOUT_PROB) ? input[idx] * scale : 0.0f;
    }
}

// ====================================================
// APPROACH 2: FUSED (Optimized)
// ====================================================

// Kernel C: ReLU aur Dropout ek saath (Register Level pe)
__global__ void fused_relu_dropout_kernel(float* input, float* output, float* mask, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 1. GLOBAL LOAD (Bus traffic start)
        float val = input[idx];
        float m = mask[idx];

        // 2. COMPUTE (Registers - Super Fast)
        float val_relu = fmaxf(0.0f, val); // ReLU
        float result = (m > DROPOUT_PROB) ? val_relu * scale : 0.0f; // Dropout

        // 3. GLOBAL STORE (Bus traffic end)
        output[idx] = result;
    }
}

// ====================================================
// MAIN BENCHMARK
// ====================================================
int main() {
    size_t bytes = N * sizeof(float);
    cout << "--- FUSED KERNEL BENCHMARK ---" << endl;
    cout << "Data Size: " << N / 1000000 << " Million Floats (" << bytes / 1024 / 1024 << " MB)" << endl;

    // --- 1. MEMORY ALLOCATION ---
    float *h_in, *h_mask;
    cudaMallocHost(&h_in, bytes);   // Pinned Memory for faster setup
    cudaMallocHost(&h_mask, bytes);

    // Initialize Random Data
    for (int i = 0; i < N; i++) {
        h_in[i] = (float)(rand() % 10 - 5); // Random floats (-5 to 5)
        h_mask[i] = (float)rand() / RAND_MAX; // Random 0.0 to 1.0
    }

    float *d_in, *d_temp, *d_out, *d_mask;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_temp, bytes); // Ye sirf Unfused ko chahiye (Memory Waste!)
    cudaMalloc(&d_out, bytes);
    cudaMalloc(&d_mask, bytes);

    // Copy Host -> Device
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, bytes, cudaMemcpyHostToDevice);

    // --- 2. SETUP ---
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float ms = 0;

    // ====================================================
    // TEST 1: NAIVE (Two Kernels + Extra Memory)
    // ====================================================
    cudaEventRecord(start);
    
    // Step 1: Read Input -> ReLU -> Write Temp
    relu_kernel<<<gridSize, blockSize>>>(d_in, d_temp, N);
    
    // Step 2: Read Temp -> Dropout -> Write Output
    dropout_kernel<<<gridSize, blockSize>>>(d_temp, d_out, d_mask, N, SCALE);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float time_naive = ms;
    cout << "\n Naive Approach (2 Kernels): " << time_naive << " ms" << endl;
    cout << "   -> Memory Traffic: 2 Reads + 2 Writes" << endl;

    // ====================================================
    // TEST 2: FUSED (One Kernel)
    // ====================================================
    cudaEventRecord(start);
    
    // Single Step: Read Input -> ReLU + Dropout -> Write Output
    fused_relu_dropout_kernel<<<gridSize, blockSize>>>(d_in, d_out, d_mask, N, SCALE);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float time_fused = ms;
    cout << "\n Fused Approach (1 Kernel):  " << time_fused << " ms" << endl;
    cout << "   -> Memory Traffic: 1 Read + 1 Write" << endl;

    // ====================================================
    // REPORT
    // ====================================================
    cout << "\n--------------------------------------------------" << endl;
    cout << "SPEEDUP: " << time_naive / time_fused << "x Faster" << endl;
    cout << "MEMORY SAVED: " << bytes / 1024 / 1024 << " MB (Intermediate buffer removed)" << endl;
    cout << "--------------------------------------------------" << endl;

    // Cleanup
    cudaFree(d_in); cudaFree(d_temp); cudaFree(d_out); cudaFree(d_mask);
    cudaFreeHost(h_in); cudaFreeHost(h_mask);

    return 0;
}