# Warp Shuffle - shuffle_sync.cu

## Purpose
Demonstrates warp-level register-to-register communication using shuffle intrinsics (`__shfl_sync`). The example shows how threads within a warp can directly exchange data without using shared memory.

---

## Problem Statement
A common pattern in GPU programming is that one lane needs a value produced by another lane in the **same warp**. Naive solutions use shared memory and synchronization, which adds overhead.

---

## Key Idea / Solution

- Each lane computes a local value (`my_value`) in a register
- A receiving lane calls `__shfl_sync(mask, my_value, src_lane)` to read the value that `src_lane` holds in its register
- This is **register-to-register** and avoids shared memory and block-wide synchronization

---

## Important Caveats About Correctness and Race Conditions

### ⚠️ Compute-before-read
Ensure the source lane computes/writes its `my_value` **before** any lane calls `__shfl_sync`. In typical cases, threads execute in lockstep within a warp, so this is naturally satisfied.

### ⚠️ Divergence
If the source lane's computation is inside a divergent branch that the source lane does not execute, the value is **undefined**.

### ⚠️ Intra-warp only
Shuffles work only within a warp (32 lanes). Cross-warp communication still requires shared/global memory.

### ⚠️ Mask usage
Use a proper mask (e.g., `0xFFFFFFFF`) to mark active lanes; otherwise behavior for inactive lanes is **undefined**.

---

## Warp Shuffle Data Flow

![Warp Shuffle Profile](OUT_Profile_shuffle_.png)

---

## How to Run

```bash
# Compile
nvcc shuffle_sync.cu -o shuffle_test

# Execute
./shuffle_test
```

---

## Expected Output

```
Thread 0: value = 0
Thread 1: value = 10
...
Thread 5: value = 50
...
After shuffle: All threads have value 50 from lane 5
```

## In detail Explanation:
## Function Signature

```cpp
T __shfl_sync(unsigned mask, T var, int srcLane, int width = warpSize)
```

### Parameters Explained

**Parameter 1: `mask` (The Gatekeeper)**
- 32-bit number where each bit represents one thread
- `1` = Thread participates in shuffle
- `0` = Thread does NOT participate
- Binary representation: `0xFFFFFFFF = 11111111 11111111 11111111 11111111`
  - Thread 31 ↑ ... Thread 0 ↑

**Parameter 2: `var` (The Data)**
- Variable to share between threads
- Supported types:
  - `int`, `unsigned int`
  - `long`, `unsigned long`
  - `long long`, `unsigned long long`
  - `float`, `double`
  - `__half`, `__half2` (with `cuda_fp16.h`)

**Parameter 3: `srcLane` (The Source)**
- Thread ID (0-31) to fetch value from
- Specifies which thread's data you want to receive

**Parameter 4: `width` (Advanced - Usually Ignored)**
- Default = 32 (full warp)
- Can subdivide warp into smaller groups
- Example: `width = 5` divides warp into groups of 16
  - Group 0: Threads 0-15 (srcLane & 16 = 0)
  - Group 1: Threads 16-31 (srcLane & 16 = 5)

---

## Common Mask Patterns

### All 32 threads active
```cpp
0xFFFFFFFF  // 11111111111111111111111111111111
// All 32 threads active
```

### Lower 16 threads only
```cpp
0x0000FFFF  // 00000000000000001111111111111111
// Threads 0-15 only
```

### Upper 16 threads only
```cpp
0xFFFF0000  // 11111111111111110000000000000000
// Threads 16-31 only
```

### Even threads only
```cpp
0x55555555  // 01010101010101010101010101010101
// Threads 0, 2, 4, 6, ... 30
```

### Odd threads only
```cpp
0xAAAAAAAA  // 10101010101010101010101010101010
// Threads 1, 3, 5, 7, ... 31
```

**Rule: Mask should match currently active threads**

---

## Usage Examples

### ✅ Valid Examples

```cpp
// Basic integer shuffle
int x = 42;
int received = __shfl_sync(0xFFFFFFFF, x, 10);
// All threads get value from thread 10

// Float shuffle
float pi = 3.14159f;
float received = __shfl_sync(0xFFFFFFFF, pi, 5);
// All threads get pi value from thread 5
```

### ❌ Invalid Examples

```cpp
// Cannot share arrays directly
int arr[10];
__shfl_sync(0xFFFFFFFF, arr, 5);  // ERROR! Can't share array

// Cannot share pointers (but can share pointer VALUES)
int *ptr;
__shfl_sync(0xFFFFFFFF, ptr, 5);  // Risky! Pointer value, not data
```

### Important Constraints
- Variable MUST be in register (local variable)
- Cannot share arrays directly
- Cannot share pointers (but can share pointer VALUES)

---

## Edge Cases & Special Scenarios

### Case 1: Invalid Source Lane
```cpp
int val = __shfl_sync(0xFFFFFFFF, my_data, 50);
// Out of range (0-31 valid)
// Result: Returns YOUR OWN value (my_data)
```

### Case 2: Source Thread Inactive
```cpp
if (lane < 14) {  // Only threads 0-13 execute
    int val = __shfl_sync(0x0000FFFF, my_data, 20);
}
// Thread 20 is INACTIVE!
// Result: UNDEFINED! (Usually 0, but don't rely on it)

// CORRECT way:
if (lane < 14) {
    int val = __shfl_sync(0x00003FFF, my_data, 5);
    // Only threads 0-13 participate ✅
    // Thread 5 is ACTIVE ✅
}
```

### Case 3: Different Values in Each Thread
```cpp
int my_id = threadIdx.x % 32;  // 6-31

// Thread 5 wants data from Thread 12
if (lane == 5) {
    int data_from_12 = __shfl_sync(0xFFFFFFFF, my_id, 12);
    printf("Got: %d\n", data_from_12);  // Prints: 12
}

// Thread 20 wants data from Thread 3
if (lane == 20) {
    int data_from_3 = __shfl_sync(0xFFFFFFFF, my_id, 3);
    printf("Got: %d\n", data_from_3);  // Prints: 3
}
```

**Important:** `srcLane` is evaluated per thread! Each thread can request from different source.

---

## Complete Working Example

### Broadcast Pattern

```cpp
__global__ void shuffle_broadcast_demo() {
    int lane = threadIdx.x % 32;  // 0-31
    
    // Step 1: Each thread has its own value
    int my_value = lane * 10;
    // Thread 0:  my_value = 0
    // Thread 1:  my_value = 10
    // Thread 5:  my_value = 50
    // Thread 10: my_value = 100
    
    // Step 2: Thread 10 has value we want to broadcast
    int received = __shfl_sync(0xFFFFFFFF, my_value, 10);
    // All threads participate
    // All threads read from Thread 10
    
    // Step 3: Result
    // SHIFT cereals main received = 100
    // Thread 0:  received = 100
    // Thread 1:  received = 100
    // Thread 15: received = 100
    // Thread 31: received = 100
    
    if (lane == 0) {
        printf("Thread 0 received: %d\n", received);  // 100
    }
    if (lane == 15) {
        printf("Thread 15 received: %d\n", received);  // 100
    }
}
```

**Visual Representation:**

```
Before shuffle:
Thread 0:  [100]
Thread 1:  [200]
...
Thread 10: [999] ← Source
Thread 31: [500]

After shuffle: int broadcasted = __shfl_sync(0xFFFFFFFF, my_value, 10);
Thread 0:  broadcasted = 999
Thread 1:  broadcasted = 999
Thread 2:  broadcasted = 999
...
Thread 10: val = 999
Thread 31: broadcasted = 999
```

---

## Why Use Shuffle Instead of Shared Memory?

### Advantages
1. **Faster** - Direct register-to-register transfer
2. **No synchronization needed** - Implicit within warp
3. **No memory allocation** - Uses registers only
4. **Lower latency** - No memory transactions

### When to Use
- Broadcasting values within a warp
- Warp-level reductions
- Data exchange between nearby threads
- Performance-critical code paths

---

## Best Practices

1. **Always specify the mask explicitly** - Don't rely on defaults
2. **Match mask to active threads** - Prevents undefined behavior
3. **Validate srcLane is in range** - And is active in mask
4. **Use with warp-level operations** - Reductions, scans, broadcasts
5. **Prefer over shared memory** - When operating within a warp

---

## Common Pitfalls

❌ **Wrong:** Using mask that doesn't match active threads
```cpp
int val = __shfl_sync(0xFFFFFFFF, data, 5);
// Hardware waits for threads 16-31 (Deadlock risk!)
```

✅ **Correct:** Match mask to active threads
```cpp
int val = __shfl_sync(0x0000FFFF, data, 5);
// Only threads 0-15 participate
```

---

---

## Next Steps

- Experiment with `__shfl_up_sync()` and `__shfl_down_sync()`
- Try different mask values
- Profile with Nsight Compute to see performance benefits
- Apply to beam search pruning kernel
