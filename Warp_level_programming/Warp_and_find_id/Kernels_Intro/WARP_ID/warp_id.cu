#include<iostream>
#include<vector>
#include<cstdio>
#include<cuda_runtime.h>
#define N 10
using namespace std;

__global__ void find_my_warp() {
    int tid=threadIdx.x;
    int warp_id=tid/32;
    int lane_id=tid%32;

    if (lane_id==0) {
        printf("Warp %d starts at thread %d\n", warp_id, tid);
    }
}

int main() {
    find_my_warp << <1,64 >> > ();
    cudaDeviceSynchronize();
    return 0;
}
