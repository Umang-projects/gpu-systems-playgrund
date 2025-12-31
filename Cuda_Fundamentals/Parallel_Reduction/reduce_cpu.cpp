#include <iostream>
#include <vector>

int main() {
    const int N = 1e7;
    std::vector<float> a(N, 1.0f);

    float sum = 0.0f;
    for (int i = 0; i < N; ++i)
        sum += a[i];

    std::cout << "Sum = " << sum << std::endl;
}
