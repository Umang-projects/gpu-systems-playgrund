# CUDA Maximum Element Finding - Naive Approach

## Problem Statement
Find the maximum element in an array of 1,000,000 integers using GPU parallelization.

## Configuration
- **Array Size**: 1,000,000 elements
- **Block Size**: 256 threads per block
- **Grid Size**: Dynamically calculated
- **Test Data**: Numbers 0-999, with maximum value 999,999 at position 500,000

## Performance Results

```
GPU Time: 0.2854 ms
Found Max: 999999
```

### Profiling Metrics (nvprof/ncu)
```
Metric Name              Metric Unit    Metric Value
DRAM Frequency                  Ghz            4.99
SM Frequency                    Mhz          585.00
Elapsed Cycles                cycle    13,469,342
Memory Throughput                 %           71.95
DRAM Throughput                   %            0.01
Duration                         ms           23.02
L1/TEX Cache Throughput           %           80.48
L2 Cache Throughput               %            0.15
SM Active Cycles             cycle   12,042,048.55
Compute (SM) Throughput           %           71.95
```

## Critical Performance Problems

### Problem 1: Atomic Serialization
**The Issue**: Every thread calls `atomicMax()` on the same global memory location, causing complete serialization of all 1,000,000 threads.

**What Happens**:
```
Thread 1: atomicMax(d_max, value1)  ← executes
Thread 2: atomicMax(d_max, value2)  ← waits for Thread 1
Thread 3: atomicMax(d_max, value3)  ← waits for Thread 2
...
Thread 1,000,000: atomicMax(d_max, valueN)  ← still waiting
```

**Impact**:
- **SM Active Cycles**: 12,042,048 cycles wasted waiting
- **Compute Throughput**: Only 71.95% (nearly 30% of GPU compute power sits idle)
- Converts parallel operation into sequential execution
- Defeats the entire purpose of using a GPU

### Problem 2: Memory Contention
**The Issue**: All threads compete for a single 4-byte memory location in global memory.

**Impact**:
- **Memory Throughput**: 71.95%
- **DRAM Throughput**: 0.01% (essentially unused despite having hundreds of GB/s bandwidth)
- Cache thrashing due to constant updates to the same location
- Memory subsystem cannot be utilized effectively

### Problem 3: Warp-Level Inefficiency
**The Issue**: Threads within the same warp (32 threads) must serialize their atomic operations.

**Impact**:
- Warps stall waiting for atomic operations to complete
- Poor warp scheduling efficiency
- Cannot hide memory latency through thread switching
- Overall GPU occupancy and utilization drops

## Why This Matters

This implementation demonstrates a common pitfall in GPU programming: **using atomic operations without considering their serialization cost**. While the code is simple and intuitive, it effectively turns the massively parallel GPU into a sequential processor for the reduction step.

The fundamental issue is that atomics on a single global variable create a bottleneck that all threads must pass through one at a time, eliminating any benefit from parallel execution.

## How to Compile and Run

```bash
# Compile
!nvcc -o kernel kernel.cu

# Run
!./kernel

# Profile with nvprof
!nvprof ./kernel
```

## Testing Environment
- **Platform**: Kaggle Notebooks
- **GPU**: NVIDIA Tesla P100
- **CUDA Version**: 11.x

## Future Work
Warp-level programming using shuffle instructions will be explored as the next optimization technique on the P100 GPU.