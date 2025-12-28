# Phase 1: GPU Programming Fundamentals

## What This Phase Is About

This is where I learned the absolute core concepts of GPU programming. These aren't fancy AI algorithms or complicated optimizations - they're the fundamental building blocks that every GPU programmer needs to understand. Think of this as learning to walk before you run.

---

## What I Built

### 1. Vector Addition - My First GPU Program

**What it does:** Adds two arrays of numbers together, element by element.

**Why it matters:** This is the "Hello World" of GPU programming. It's simple, but it taught me how GPUs actually work at a low level.

**What I learned:**
- **Thread Indexing**: How to map thousands of threads to array elements. Each thread needs to know "which piece of data am I responsible for?"
- **Memory Coalescing**: Why the way threads access memory makes a HUGE difference in performance. If threads access memory incorrectly, you can lose 80% of your speed.
- **Launch Configuration**: How to decide how many threads to use and how to organize them into blocks.
- **Error Handling**: How to catch and debug GPU errors (because trust me, things go wrong).

**What I proved:** I understand how threads work on a GPU and how to write efficient memory access patterns.

**Performance:** Achieved ~450 GB/s memory bandwidth (close to the theoretical maximum of my GPU).

---

### 2. Matrix Multiplication - Understanding the Memory Hierarchy

**What it does:** Multiplies two matrices together (the fundamental operation in machine learning and graphics).

**Why it matters:** This is where you really learn about GPU memory. A naive implementation is **100x slower** than an optimized one, and the only difference is how you use memory.

**What I learned:**
- **The Memory Hierarchy Problem**: Global memory is slow (like going to a warehouse), shared memory is fast (like having a desk drawer), registers are fastest (like keeping things in your hand).
- **Tiling Strategy**: Instead of loading the same data from slow memory over and over, load it once into fast shared memory and reuse it.
- **Register Blocking**: Even faster than shared memory - keep frequently used values in registers.
- **Arithmetic Intensity**: The ratio of computation to memory access. Good GPU programs do lots of math per memory fetch.

**What I proved:** I can identify memory bottlenecks and fix them using the GPU memory hierarchy.

**Performance Journey:**
- **Naive version**: 45 GFLOPS (terrible)
- **Shared memory tiling**: 380 GFLOPS (8× faster!)
- **Register blocking**: 720 GFLOPS (16× faster than naive!)

This progression shows I didn't just copy code - I understand WHY each optimization works.

---

### 3. Parallel Reduction - Synchronization and Warp Programming

**What it does:** Adds up all numbers in a large array to produce a single sum.

**Why it matters:** This taught me how threads communicate and coordinate with each other.

**What I learned:**
- **The Atomic Problem**: If every thread tries to update the same variable, they all wait in line (completely defeating the purpose of parallelism).
- **Tree Reduction**: How to structure parallel algorithms so threads work together efficiently.
- **Synchronization**: When and why you need `__syncthreads()` to make sure threads don't race ahead of each other.
- **Warp-Level Programming**: Threads within a warp (32 threads) can communicate without expensive synchronization using shuffle instructions.
- **Bank Conflicts**: How shared memory is organized into banks and why accessing the same bank from multiple threads causes slowdowns.

**What I proved:** I understand thread synchronization, race conditions, and warp-level optimization.

**Performance Journey:**
- **Atomic operations**: 0.8 ms (serialized, terrible)
- **Shared memory reduction**: 0.15 ms (5× faster)
- **Warp shuffle reduction**: 0.09 ms (9× faster than atomics)

---

## The Big Picture: What These Projects Prove

### I Understand the GPU Memory Hierarchy

Most performance problems on GPUs are memory problems, not compute problems. These projects show I can:
- Identify when a program is memory-bound
- Use shared memory to reduce global memory traffic
- Use registers for maximum performance
- Measure and optimize memory bandwidth utilization

### I Understand Thread Organization and Synchronization

GPUs have thousands of threads running at once. These projects show I can:
- Properly index threads to avoid conflicts
- Synchronize threads when necessary (but not unnecessarily)
- Use warp-level primitives for fine-grained control
- Avoid common pitfalls like race conditions and deadlocks

### I Can Measure and Optimize Performance

For every project, I:
- Started with a CPU baseline to know what "fast enough" looks like
- Built a naive GPU version to establish a baseline
- Identified bottlenecks using profiling tools (Nsight Compute)
- Applied targeted optimizations
- Measured the impact of each change with real data

This isn't theoretical knowledge - I have the performance numbers to prove it works.

---

## Tools I Used

- **CUDA**: NVIDIA's parallel computing platform
- **Nsight Compute**: GPU profiler that shows memory bandwidth, occupancy, warp efficiency
- **Nsight Systems**: Timeline view of GPU execution
- **nvcc**: NVIDIA's CUDA compiler

---

## Why These Projects Matter for Real Work

**Vector Addition** → Data preprocessing, element-wise operations in neural networks

**Matrix Multiplication** → Core of deep learning (every neural network layer), computer graphics, scientific computing

**Parallel Reduction** → Computing statistics (mean, variance), loss functions, gradient norms

These aren't toy problems - they're the building blocks of real GPU applications. Deep learning frameworks like PyTorch and TensorFlow are built on these same primitives.

---

## What's Next

Now that I understand the fundamentals, I'm ready to:
- Optimize real ML kernels (Flash Attention, fused operations)
- Work with more complex memory patterns (convolution, histograms)
- Build complete GPU-accelerated applications

But everything builds on these foundations. You can't write fast GPU code without understanding memory hierarchy, thread synchronization, and performance measurement.

---

## Key Takeaways

1. **Memory is the bottleneck**: Most GPU code is limited by memory bandwidth, not compute power
2. **Measure everything**: You can't optimize what you don't measure
3. **Start simple, iterate**: Naive → Shared Memory → Registers is a proven optimization path
4. **Understand the hardware**: Warps, memory banks, coalescing - these aren't optional knowledge

---

## Running the Code

Each project includes:
- `README.md` - Detailed explanation of the problem and optimizations
- CPU baseline implementation
- Multiple GPU versions showing progression
- Benchmark code with performance measurements
- Profiling data showing bottlenecks

See individual project directories for build instructions and detailed performance analysis.