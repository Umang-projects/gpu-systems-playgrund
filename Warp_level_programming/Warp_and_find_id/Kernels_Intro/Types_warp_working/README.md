# CUDA Shuffle Functions - Theory Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Why Use Shuffle Functions?](#why-use-shuffle-functions)
3. [Basic Concepts](#basic-concepts)
4. [Types of Shuffle Functions](#types-of-shuffle-functions)
   - [__shfl_sync()](#1-__shfl_sync---direct-shuffle)
   - [__shfl_up_sync()](#2-__shfl_up_sync---shift-left)
   - [__shfl_down_sync()](#3-__shfl_down_sync---shift-right)
   - [__shfl_xor_sync()](#4-__shfl_xor_sync---butterfly-pattern)
5. [Common Parameters](#common-parameters)
6. [Data Types Supported](#data-types-supported)
7. [Performance Characteristics](#performance-characteristics)
8. [Common Use Cases](#common-use-cases)
9. [Best Practices](#best-practices)

---

## Introduction

CUDA shuffle functions are **warp-level primitives** that allow threads within a warp (32 threads) to exchange data directly through registers, without using shared memory or global memory.

### What is a Warp?
- A warp is a group of **32 threads** that execute together
- Threads in a warp are numbered from **0 to 31** (called lane ID)
- All threads in a warp execute the same instruction simultaneously (SIMT model)

### Key Benefits
- ⚡ **Extremely fast** - Direct register-to-register communication
- 🚀 **No synchronization** - Implicit synchronization within warp
- 💾 **No memory overhead** - Doesn't use shared or global memory
- 🎯 **Simple to use** - Single function call

---

## Why Use Shuffle Functions?

### Traditional Approach (Shared Memory)
```
1. Write data to shared memory
2. __syncthreads() - Wait for all threads
3. Read data from shared memory
4. __syncthreads() - Wait again

Total: ~40-60 clock cycles
```

### Shuffle Approach
```
1. __shfl_sync() - Direct exchange

Total: ~1-2 clock cycles
```

### Speed Comparison
| Method | Latency | Memory Used |
|--------|---------|-------------|
| Shuffle Functions | 1-2 cycles | 0 bytes |
| Shared Memory | 20-40 cycles | Requires allocation |
| Global Memory | 200-400 cycles | Requires allocation |

**Conclusion:** Shuffle functions are **10-20x faster** than shared memory!

---

## Basic Concepts

### Lane ID
Every thread in a warp has a unique ID called **lane ID** (0-31):
```cpp
int lane = threadIdx.x % 32;  // Gets lane ID (0-31)
```

### Mask Parameter
The mask is a 32-bit value where each bit represents whether a thread participates:
- Bit = `1` → Thread participates ✅
- Bit = `0` → Thread does NOT participate ❌

**Common masks:**
```cpp
0xFFFFFFFF  // All 32 threads (most common)
0x0000FFFF  // Threads 0-15 only
0xFFFF0000  // Threads 16-31 only
0x55555555  // Even threads (0,2,4,...)
0xAAAAAAAA  // Odd threads (1,3,5,...)
```

**Important:** Mask should match the threads that are actually active!

### Source and Destination
- **Source thread** - Thread that provides the data
- **Destination thread** - Thread that receives the data
- Each thread can read from any other thread in the warp

---

## Types of Shuffle Functions

## 1. `__shfl_sync()` - Direct Shuffle

### Signature
```cpp
T __shfl_sync(unsigned mask, T var, int srcLane, int width = 32)
```

### Purpose
Gets data from **one specific thread** you choose.

### How It Works
- Every thread specifies which thread to read from
- All threads can read from different sources
- Like a broadcast when all threads read from the same source

### Visual Model
```
Threads:    0    1    2    3    4    5
Values:    [A]  [B]  [C]  [D]  [E]  [F]

After __shfl_sync(mask, var, 3):
All threads get value from thread 3

Results:   [D]  [D]  [D]  [D]  [D]  [D]
```

### Communication Pattern
- **One-to-Many** - One source, multiple destinations
- **Broadcast** - Sharing one value to all threads
- **Random Access** - Each thread picks different source

### Parameters
- `mask` - Which threads participate
- `var` - The value to share
- `srcLane` - Which thread to read from (0-31)
- `width` - Warp subdivision (usually 32)

### When to Use
✅ Broadcasting a value to all threads  
✅ Leader election patterns  
✅ Getting data from a specific thread  
✅ Conditional data gathering  

### Edge Cases
- If `srcLane` is out of range (>31) → Returns your own value
- If source thread is inactive → Result is UNDEFINED
- If source thread doesn't exist → Returns your own value

---

## 2. `__shfl_up_sync()` - Shift Left

### Signature
```cpp
T __shfl_up_sync(unsigned mask, T var, unsigned int delta, int width = 32)
```

### Purpose
Gets data from a thread that is **ABOVE** (lower lane ID) you.

### How It Works
- Shifts data "upward" in lane numbering
- Each thread reads from: `myLane - delta`
- Threads at the top get their own value (nowhere to read from)

### Visual Model
```
Before:
Threads:  0    1    2    3    4    5
Values:  [A]  [B]  [C]  [D]  [E]  [F]

After __shfl_up_sync(mask, var, 2):
Each thread reads from 2 positions above

Thread 0: myLane(0) - 2 = -2 (invalid) → keeps [A]
Thread 1: myLane(1) - 2 = -1 (invalid) → keeps [B]
Thread 2: myLane(2) - 2 = 0 → gets [A]
Thread 3: myLane(3) - 2 = 1 → gets [B]
Thread 4: myLane(4) - 2 = 2 → gets [C]
Thread 5: myLane(5) - 2 = 3 → gets [D]

Results:  [A]  [B]  [A]  [B]  [C]  [D]
```

### Communication Pattern
- **Shift Left** - Data flows from lower to higher lanes
- **Sequential Dependencies** - Each thread depends on previous
- **Prefix Operations** - Building cumulative results

### Parameters
- `mask` - Which threads participate
- `var` - The value to share
- `delta` - How many positions above (1 = immediate neighbor)
- `width` - Warp subdivision (usually 32)

### When to Use
✅ Prefix sum (scan) operations  
✅ Cumulative operations  
✅ Data dependencies on previous elements  
✅ Sliding window (looking backward)  

### Edge Cases
- Threads at positions 0 to (delta-1) get their own values
- Cannot read from negative lane IDs

---

## 3. `__shfl_down_sync()` - Shift Right

### Signature
```cpp
T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width = 32)
```

### Purpose
Gets data from a thread that is **BELOW** (higher lane ID) you.

### How It Works
- Shifts data "downward" in lane numbering
- Each thread reads from: `myLane + delta`
- Threads at the bottom get their own value (nowhere to read from)

### Visual Model
```
Before:
Threads:  0    1    2    3    4    5
Values:  [A]  [B]  [C]  [D]  [E]  [F]

After __shfl_down_sync(mask, var, 2):
Each thread reads from 2 positions below

Thread 0: myLane(0) + 2 = 2 → gets [C]
Thread 1: myLane(1) + 2 = 3 → gets [D]
Thread 2: myLane(2) + 2 = 4 → gets [E]
Thread 3: myLane(3) + 2 = 5 → gets [F]
Thread 4: myLane(4) + 2 = 6 (invalid) → keeps [E]
Thread 5: myLane(5) + 2 = 7 (invalid) → keeps [F]

Results:  [C]  [D]  [E]  [F]  [E]  [F]
```

### Communication Pattern
- **Shift Right** - Data flows from higher to lower lanes
- **Reduction Trees** - Combining results from multiple threads
- **Reverse Sequential** - Processing in reverse order

### Parameters
- `mask` - Which threads participate
- `var` - The value to share
- `delta` - How many positions below (1 = immediate neighbor)
- `width` - Warp subdivision (usually 32)

### When to Use
✅ Reduction operations (sum, max, min)  
✅ Tree-based algorithms  
✅ Propagating values downward  
✅ Reverse sliding window  

### Edge Cases
- Threads at positions (32-delta) to 31 get their own values
- Cannot read from lane IDs beyond 31

---

## 4. `__shfl_xor_sync()` - Butterfly Pattern

### Signature
```cpp
T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width = 32)
```

### Purpose
Gets data from a thread calculated by **XOR operation**: `source = myLane XOR laneMask`

### How It Works
- Uses bitwise XOR to determine partner thread
- Creates symmetrical "butterfly" communication patterns
- Threads swap data with their XOR partners

### XOR Operation Explained
```
XOR Truth Table:
0 XOR 0 = 0
0 XOR 1 = 1
1 XOR 0 = 1
1 XOR 1 = 0

Examples:
Thread 0 (binary: 0000) XOR 1 (0001) = Thread 1 (0001)
Thread 1 (binary: 0001) XOR 1 (0001) = Thread 0 (0000)
Thread 5 (binary: 0101) XOR 1 (0001) = Thread 4 (0100)

Notice: XOR is symmetric!
If A XOR mask = B, then B XOR mask = A
```

### Visual Model - XOR with 1
```
Before:
Threads:  0    1    2    3    4    5    6    7
Values:  [A]  [B]  [C]  [D]  [E]  [F]  [G]  [H]

After __shfl_xor_sync(mask, var, 1):
Each thread XORs with 1 to find partner

Thread 0: 0 XOR 1 = 1 → gets [B]
Thread 1: 1 XOR 1 = 0 → gets [A]
Thread 2: 2 XOR 1 = 3 → gets [D]
Thread 3: 3 XOR 1 = 2 → gets [C]
Thread 4: 4 XOR 1 = 5 → gets [F]
Thread 5: 5 XOR 1 = 4 → gets [E]
Thread 6: 6 XOR 1 = 7 → gets [H]
Thread 7: 7 XOR 1 = 6 → gets [G]

Results:  [B]  [A]  [D]  [C]  [F]  [E]  [H]  [G]

Pattern: Neighbors swap with each other!
```

### Visual Model - XOR with 2
```
Before:
Threads:  0    1    2    3    4    5    6    7
Values:  [A]  [B]  [C]  [D]  [E]  [F]  [G]  [H]

After __shfl_xor_sync(mask, var, 2):

Thread 0: 0 XOR 2 = 2 → gets [C]
Thread 1: 1 XOR 2 = 3 → gets [D]
Thread 2: 2 XOR 2 = 0 → gets [A]
Thread 3: 3 XOR 2 = 1 → gets [B]
Thread 4: 4 XOR 2 = 6 → gets [G]
Thread 5: 5 XOR 2 = 7 → gets [H]
Thread 6: 6 XOR 2 = 4 → gets [E]
Thread 7: 7 XOR 2 = 2 → gets [F]

Results:  [C]  [D]  [A]  [B]  [G]  [H]  [E]  [F]

Pattern: Pairs 2 positions apart swap!
```

### Butterfly Pattern Visualization
```
Using powers of 2 creates butterfly pattern:

laneMask = 1:   0↔1,  2↔3,  4↔5,  6↔7  (distance 1)
laneMask = 2:   0↔2,  1↔3,  4↔6,  5↔7  (distance 2)
laneMask = 4:   0↔4,  1↔5,  2↔6,  3↔7  (distance 4)
laneMask = 8:   0↔8,  1↔9,  2↔10, 3↔11 (distance 8)
laneMask = 16:  0↔16, 1↔17, 2↔18, 3↔19 (distance 16)

This creates a logarithmic communication pattern!
```

### Communication Pattern
- **Butterfly Exchange** - Symmetrical partner swapping
- **Logarithmic Communication** - O(log N) operations
- **Parallel Reduction** - All threads contribute equally

### Parameters
- `mask` - Which threads participate
- `var` - The value to share
- `laneMask` - XOR value to compute partner (usually power of 2)
- `width` - Warp subdivision (usually 32)

### When to Use
✅ Fast parallel reductions (sum, max, min)  
✅ Butterfly networks  
✅ FFT (Fast Fourier Transform)  
✅ Algorithms requiring symmetrical exchange  
✅ Logarithmic-time operations  

### Why XOR is Powerful
- **Symmetrical** - If A reads from B, then B reads from A
- **Self-inverse** - Applying XOR twice returns to original
- **Logarithmic depth** - Only log₂(32) = 5 steps for full warp
- **No conflicts** - Each thread has unique partner

### Edge Cases
- XOR result is always within 0-31 for warp operations
- Always produces valid lane ID (no boundary issues)

---

## Common Parameters

### 1. Mask (`unsigned mask`)

**Purpose:** Controls which threads participate in the shuffle operation.

**Format:** 32-bit integer where each bit represents one thread:
```
Bit position: 31 30 29 ... 2  1  0
Thread ID:    31 30 29 ... 2  1  0
```

**Common Values:**
```cpp
0xFFFFFFFF  // All threads (binary: 11111111...11111111)
0x0000FFFF  // Lower 16 threads (binary: 00000000...11111111)
0xFFFF0000  // Upper 16 threads (binary: 11111111...00000000)
0x00000001  // Only thread 0
0x80000000  // Only thread 31
```

**Important Rules:**
- Mask MUST match currently active threads
- Inactive threads in mask → Undefined behavior
- Using wrong mask → Can cause deadlocks

---

### 2. Variable (`T var`)

**Purpose:** The data value to be exchanged.

**Type Requirements:**
- Must be a **register variable** (local variable)
- Must be a supported data type
- Cannot be array or pointer

**Size Limits:**
- Maximum 32 bits for basic types
- 64 bits for `double` and `long long`

---

### 3. Width (`int width`)

**Purpose:** Subdivides the warp into smaller groups.

**Default:** 32 (full warp)

**How It Works:**
```
width = 32: All 32 threads in one group
width = 16: Two groups (0-15, 16-31)
width = 8:  Four groups (0-7, 8-15, 16-23, 24-31)
```

**Usage:**
- Threads can only exchange data within their group
- Rarely used in practice
- Most algorithms use default (32)

**Example:**
```cpp
// With width=16, thread 5 can only read from threads 0-15
int val = __shfl_sync(0xFFFFFFFF, data, 20, 16);
// Thread 5 actually reads from thread 4 (20 % 16 = 4)
```

---

## Data Types Supported

### ✅ Supported Types

**Integer Types:**
```cpp
int
unsigned int
long
unsigned long
long long
unsigned long long
```

**Floating Point Types:**
```cpp
float
double
```

**Half Precision (with cuda_fp16.h):**
```cpp
__half
__half2
```

### ❌ Unsupported Types

**Arrays:**
```cpp
int array[10];  // ❌ Cannot shuffle arrays
```

**Pointers:**
```cpp
int *ptr;  // ❌ Can shuffle pointer VALUE, not data
```

**Structures:**
```cpp
struct MyStruct { int a; float b; };  // ❌ Cannot shuffle directly
```

**Workaround for Complex Types:**
```cpp
// Shuffle members individually
struct Data {
    int x;
    float y;
};

Data d;
d.x = __shfl_sync(0xFFFFFFFF, d.x, 0);
d.y = __shfl_sync(0xFFFFFFFF, d.y, 0);
```

---

## Performance Characteristics

### Latency Comparison

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Shuffle Functions | 1-2 cycles | Very High |
| Shared Memory | 20-40 cycles | High |
| L1 Cache | 30-80 cycles | Medium |
| L2 Cache | 200-300 cycles | Low |
| Global Memory | 400-800 cycles | Very Low |

### Why Shuffle is Fast

1. **Register-to-Register** - No memory hierarchy involved
2. **Hardware Accelerated** - Dedicated hardware for shuffles
3. **No Synchronization** - Implicit warp synchronization
4. **Zero Memory Footprint** - Doesn't consume shared/global memory

### Performance Best Practices

✅ **Use shuffle for warp-level operations**  
✅ **Prefer shuffle over shared memory when possible**  
✅ **Combine multiple shuffles efficiently**  
✅ **Use XOR for symmetrical operations**  
✅ **Avoid unnecessary type conversions**  

❌ **Don't use shuffle across warps**  
❌ **Don't shuffle in divergent control flow unnecessarily**  
❌ **Don't assume specific ordering beyond documentation**  

---

## Common Use Cases

### 1. Broadcasting
**Problem:** One thread has data, all threads need it.

**Solution:** Use `__shfl_sync()`

**Pattern:**
```
Thread 0 computes important value
All threads read from thread 0
```

---

### 2. Prefix Sum (Scan)
**Problem:** Compute cumulative sum up to each position.

**Solution:** Use `__shfl_up_sync()` with doubling offsets

**Pattern:**
```
Offset 1:  Add from 1 position back
Offset 2:  Add from 2 positions back
Offset 4:  Add from 4 positions back
Offset 8:  Add from 8 positions back
Offset 16: Add from 16 positions back
```

**Algorithm:** Hillis-Steele parallel scan

---

### 3. Reduction (Sum/Max/Min)
**Problem:** Combine values from all threads into one result.

**Solution:** Use `__shfl_down_sync()` or `__shfl_xor_sync()`

**Pattern with down:**
```
Offset 16: Combine pairs 16 apart
Offset 8:  Combine pairs 8 apart
Offset 4:  Combine pairs 4 apart
Offset 2:  Combine pairs 2 apart
Offset 1:  Combine pairs 1 apart
```

**Pattern with XOR:**
```
Mask 16: Combine XOR pairs (distance 16)
Mask 8:  Combine XOR pairs (distance 8)
Mask 4:  Combine XOR pairs (distance 4)
Mask 2:  Combine XOR pairs (distance 2)
Mask 1:  Combine XOR pairs (distance 1)
```

**Algorithm:** Tree reduction in O(log N)

---

### 4. Data Rotation
**Problem:** Rotate data circularly within warp.

**Solution:** Use `__shfl_sync()` with calculated indices

**Pattern:**
```
Each thread reads from (myLane + rotation) % 32
```

---

### 5. Matrix Transpose (Small Tiles)
**Problem:** Transpose small matrix tiles efficiently.

**Solution:** Use combination of shuffles

**Pattern:**
```
Use XOR shuffle for butterfly exchanges
Combine with up/down for row/column ops
```

---

### 6. Parallel Sorting (Bitonic Sort)
**Problem:** Sort data within warp.

**Solution:** Use `__shfl_xor_sync()` for compare-exchange

**Pattern:**
```
Butterfly compare-exchange at each level
XOR determines comparison partners
```

---

## Best Practices

### ✅ DO:

1. **Match mask to active threads**
   - Always set mask based on which threads are running
   - Use `0xFFFFFFFF` only when all 32 threads are active

2. **Validate source lane**
   - Ensure srcLane is within 0-31
   - Check that source thread is active

3. **Use appropriate shuffle type**
   - Direct access → `__shfl_sync()`
   - Sequential dependencies → `__shfl_up_sync()`
   - Reductions → `__shfl_down_sync()` or `__shfl_xor_sync()`

4. **Leverage XOR for symmetry**
   - Use for parallel reductions
   - Exploit logarithmic communication

5. **Keep operations within warp**
   - Don't try to shuffle across warps
   - Use shared memory for inter-warp communication

6. **Profile and optimize**
   - Measure actual performance
   - Consider register pressure

### ❌ DON'T:

1. **Don't use wrong mask**
   ```cpp
   // BAD: Only 16 threads active but mask says 32
   if (lane < 16) {
       val = __shfl_sync(0xFFFFFFFF, data, 5);
   }
   ```

2. **Don't shuffle unsupported types**
   ```cpp
   // BAD: Cannot shuffle arrays
   int arr[10];
   __shfl_sync(0xFFFFFFFF, arr, 0);
   ```

3. **Don't ignore boundary conditions**
   ```cpp
   // BAD: Thread 0 has no thread above
   val = __shfl_up_sync(0xFFFFFFFF, data, 1);
   // Need to check: if (lane >= 1) before using
   ```

4. **Don't assume ordering**
   - Shuffle doesn't guarantee specific execution order
   - Don't rely on side effects

5. **Don't use in divergent branches unnecessarily**
   - All threads in mask must execute shuffle
   - Divergence can reduce efficiency

6. **Don't forget synchronization across warps**
   - Shuffle only synchronizes within warp
   - Still need `__syncthreads()` between warps

---

## Common Pitfalls and Solutions

### Pitfall 1: Inactive Source Thread
**Problem:**
```cpp
if (lane < 16) {
    val = __shfl_sync(0xFFFFFFFF, data, 20);  // Thread 20 not active!
}
```

**Solution:**
```cpp
if (lane < 16) {
    val = __shfl_sync(0x0000FFFF, data, 5);  // Thread 5 is active
}
```

---

### Pitfall 2: Wrong Offset Direction
**Problem:**
```cpp
// Want previous thread but using down
val = __shfl_down_sync(0xFFFFFFFF, data, 1);
```

**Solution:**
```cpp
// Use up for previous thread
val = __shfl_up_sync(0xFFFFFFFF, data, 1);
```

---

### Pitfall 3: Forgetting Boundary Checks
**Problem:**
```cpp
int prev = __shfl_up_sync(0xFFFFFFFF, data, 1);
result = data - prev;  // Thread 0 gets wrong answer!
```

**Solution:**
```cpp
int prev = __shfl_up_sync(0xFFFFFFFF, data, 1);
if (lane >= 1) {
    result = data - prev;
} else {
    result = 0;  // Handle boundary
}
```

---

### Pitfall 4: Using Non-Power-of-2 XOR Mask
**Problem:**
```cpp
val = __shfl_xor_sync(0xFFFFFFFF, data, 3);  // 3 is not power of 2
```

**Solution:**
```cpp
// Use powers of 2 for clean patterns
val = __shfl_xor_sync(0xFFFFFFFF, data, 4);  // 4 = 2^2
```

---

## Summary Comparison Table

| Function | Direction | Pattern | Best For | Complexity |
|----------|-----------|---------|----------|------------|
| `__shfl_sync()` | Any → Any | Broadcast | Random access, broadcasting | O(1) |
| `__shfl_up_sync()` | ← Left | Shift left | Prefix sum, cumulative ops | O(log N) for scan |
| `__shfl_down_sync()` | → Right | Shift right | Reductions, tree algorithms | O(log N) for reduction |
| `__shfl_xor_sync()` | ↔ Butterfly | Butterfly | Fast reductions, FFT | O(log N) for reduction |

---

## Key Takeaways

1. **Shuffle functions are extremely fast** - 10-20x faster than shared memory.

2. **Four types serve different purposes** - Choose the right one for your pattern.

3. **Mask must match active threads** - Critical for correctness.

4. **Works only within a warp** - Cannot exchange data across warps.

5. **XOR creates efficient patterns** - Use for symmetrical operations.

6. **Always validate boundaries** - Handle edge cases at warp boundaries

---

## Further Reading

- **NVIDIA CUDA Programming Guide** - Official documentation
- **Warp Shuffle Functions** - Detailed API reference
- **CUDA Best Practices Guide** - Optimization techniques
- **Parallel Algorithms** - Understanding parallel patterns

---