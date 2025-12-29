//%%writefile vector_add_cpu.cpp
#include <iostream>
#include <vector>
#include <chrono> // Time measure karne ke liye

using namespace std;

// CPU function vector addition ke liye
void vectorAddCPU(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int N = 1 << 26; //(approx 67 million elements)
    size_t bytes = N * sizeof(float);

    // Vectors allocate
    vector<float> h_a(N);
    vector<float> h_b(N);
    vector<float> h_c(N);

    // Data initialize
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // Timer start
    auto start = chrono::high_resolution_clock::now();

    // Addition perform
    vectorAddCPU(h_a.data(), h_b.data(), h_c.data(), N);

    // Timer stop
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration = end - start;

    cout << "CPU Time taken: " << duration.count() << " ms" << endl;

    return 0;
}