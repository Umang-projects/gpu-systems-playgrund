#include <stdio.h>

__global__ void shift_down_example(){
    int lane=threadIdx.x%32;
    int my_value=lane;
    
    // Get value from 1 position BELOW (next thread)
    int from_below = __shfl_down_sync(0xffffffff,my_value,1);
    
    if (lane < 31) {  // Last thread ko skip karo kyunki uske neeche koi nahi
        printf("Thread %d: my_value=%d, from_below=%d\n",lane,my_value,from_below);
    }
}

int main() {
    shift_down_example<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}