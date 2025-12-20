#include<iostream>
#include<vector>
#include<cstdio>
#include<cuda_runtime.h>
#define N 10
using namespace std;

__global__ void find_my_warp() {
    int tid=threadIdx.x;
    int lane_id=tid%32;
    //lets suppose 10's have 250 value.
    int my_10s_value = 10 * 25;


    //thread 5 wants thread 10's value.
    if (lane_id==5) {
        int Value_of_10 = __shfl_sync(0xFFFFFFFF, my_10s_value, 10);
        printf("Thread 5 got value from thread 10: %d\n", Value_of_10);
    }
}

int main() {
    find_my_warp << <1,64 >> > ();
    cudaDeviceSynchronize();
    return 0;
}
