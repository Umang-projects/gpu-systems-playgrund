#include <stdio.h>

__global__ void shift_xor_example(){
    int lane=threadIdx.x%32;
    int my_value=lane;
    
    // XOR with 1: threads swap values with their neighbor
    // Thread 0 <-> Thread 1
    // Thread 2 <-> Thread 3
    // Thread 4 <-> Thread 5
    int neighbor_value = __shfl_xor_sync(0xffffffff, my_value, 1);
    
    printf("Thread %d: my_value=%d, neighbor_value=%d\n", 
           lane, my_value, neighbor_value);
}

int main() {
    shift_xor_example<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}



/*
**Simple explanation:**

**__shfl_xor_sync` kya karta hai:**
- Yeh XOR operation use karke threads ko pair bana deta hai
- Jaise dance mein partner swap karte hain!

**XOR with 1 (binary):**
- Thread 0 (binary: 00000) XOR 1 = Thread 1 (binary: 00001)
- Thread 1 (binary: 00001) XOR 1 = Thread 0 (binary: 00000)
- Thread 2 (binary: 00010) XOR 1 = Thread 3 (binary: 00011)
- Thread 3 (binary: 00011) XOR 1 = Thread 2 (binary: 00010)

**Output:**
Thread 0: my_value=0, neighbor_value=1
Thread 1: my_value=1, neighbor_value=0
Thread 2: my_value=2, neighbor_value=3
Thread 3: my_value=3, neighbor_value=2
Thread 4: my_value=4, neighbor_value=5
Thread 5: my_value=5, neighbor_value=4

*/