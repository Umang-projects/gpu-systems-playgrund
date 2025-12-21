#include <stdio.h>

__global__ void shift_up_example() {
    int lane=threadIdx.x%32;
    int my_value=lane;
    
    // Get value from 1 position above (previous thread)
    int from_above=__shfl_up_sync(0xFFFFFFFF,my_value,1);
    
    if (lane < 8) {
        printf("Thread %d: my_value=%d, from_above=%d\n",lane,my_value,from_above);
    }
}

int main() {
    shift_up_example<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}