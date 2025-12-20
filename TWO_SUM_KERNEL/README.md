# Two Sum on GPU (CUDA) — Learning Project
### Question= Given an array of integers and a target, return indices of two numbers that add up to the target.
## What is this project?
This project implements a **brute-force Two Sum** algorithm on the GPU using CUDA.

The purpose of this code is **not** to build the fastest Two Sum solution.  
Instead, it is a **learning and exploration project** to understand:

- How CUDA kernels are written
- How threads work in parallel
- How atomics are used to avoid race conditions
- Why GPU performance behaves very differently from CPU performance

---

## Why Two Sum?
Two Sum is a very simple problem, which makes it perfect for learning GPU behavior.

Because the algorithm itself is easy to understand, we can focus on:
- Thread execution
- Memory access patterns
- Early-exit behavior
- Atomic operations
- Performance differences between CPU and GPU

---

## How the GPU kernel works
- Each GPU thread is responsible for one index `i`
- The thread checks all pairs `(i, j)` where `j > i`
- If `nums[i] + nums[j] == target`, the thread tries to write the result
- An `atomicExch` flag ensures that **only one thread writes the answer**
- Other threads may continue running due to GPU execution model

This design intentionally shows how **early exit is not instant on GPUs**.

---

## Important learning points

### 1. Early exit on GPU is different from CPU
On CPU, once a solution is found, we can stop immediately.

On GPU:
- Threads run in groups (warps)
- Even after a match is found, other threads may still execute
- Early exit does not instantly stop the entire kernel

---

### 2. This kernel is memory-heavy
The brute-force approach causes many global memory reads.

As a result:
- Performance is dominated by memory access
- The kernel is not compute-bound
- This explains why GPU performance may not scale linearly

---

### 3. Data-dependent performance
The runtime changes depending on where the matching pair exists:
- Early match → faster
- Late match → slower

Even with the same input size, performance varies.

---

## What this project is NOT
- This is **not** an optimized Two Sum implementation
- This is **not** a production-ready algorithm
- This is **not** meant to beat CPU hash-based solutions

---

## Why this project is useful
This project helps build intuition about:
- CUDA thread execution
- Atomic operations
- GPU memory behavior
- Why some algorithms are poor fits for GPUs
- How to reason about GPU performance before optimizing

## Problem: In GPU programming, when many threads try to update the same memory location using atomic operations, performance degrades because these operations become serialized and cause memory contention. Although atomic instructions guarantee correctness, they introduce high latency, especially when executed by a large number of threads.

## Solution: Warp-level programming

---

## How to build and run

```bash
nvcc -O3 two_sum.cu -o two_sum
./two_sum
