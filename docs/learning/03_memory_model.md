# Level 2b: Memory Model Deep Dive

**Reading Time: 1.5 hours**

Imagine your computer's GPU memory is like a warehouse. You have a limited number of shelves (memory), and you need to store boxes (tensors) efficiently. The biggest challenge in LLM inference isn't making computations faster—it's organizing the warehouse so workers (GPU threads) don't waste time looking for boxes or waiting for the forklift (memory bus). This tutorial teaches you exactly how TensorRT-Edge-LLM organizes its warehouse.

## The Core Insight First

1. **LLM inference is memory-bound** - The GPU can compute faster than memory can deliver data
2. **Pre-allocation eliminates latency spikes** - All memory is reserved upfront, no runtime allocation
3. **The KV cache is 70%+ of memory usage** - Understanding it is understanding the system

## Key Numbers to Memorize

| Metric | Value | Formula/Context |
|--------|-------|-----------------|
| **GPU memory bandwidth** | ~900 GB/s (Orin) | Theoretical max data transfer rate |
| **FP16 element size** | 2 bytes | Half-precision floating point |
| **FP8 element size** | 1 byte | Quarter precision |
| **KV cache per token per layer** | hidden_size × 4 bytes | K and V, each hidden_size × 2 bytes |
| **7B model KV cache at 4K** | ~500MB | 32 layers × 4K tokens × 1MB/1K tokens |
| **cudaMalloc latency** | ~1ms | Why we pre-allocate everything |

## Table of Contents

1. [GPU Memory Hierarchy](#1-gpu-memory-hierarchy) (20 min)
2. [The Tensor Class](#2-the-tensor-class) (20 min)
3. [KV Cache Architecture](#3-kv-cache-architecture) (30 min)
4. [Memory Pre-allocation Strategy](#4-memory-pre-allocation-strategy) (20 min)

---

## 1. GPU Memory Hierarchy

### The Memory Pyramid

```
                        ┌───────────┐
                        │ Registers │  Fastest
                        │  64KB     │  ~1 cycle access
                        └─────┬─────┘
                              │
                        ┌─────▼─────┐
                        │  Shared   │
                        │  Memory   │  ~20 cycle access
                        │  96KB     │  Shared within block
                        └─────┬─────┘
                              │
                        ┌─────▼─────┐
                        │ L2 Cache  │
                        │  4MB      │  ~200 cycle access
                        │           │  Shared across GPU
                        └─────┬─────┘
                              │
                   ┌──────────▼──────────┐
                   │    Global Memory    │  Slowest
                   │    (HBM/GDDR)       │  ~400 cycle access
                   │    32GB (Orin)      │  Where KV cache lives
                   └─────────────────────┘
```

### Why Memory Bandwidth Matters

```
┌────────────────────────────────────────────────────────────────────────┐
│                    THE ROOFLINE MODEL                                  │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  FLOPS  ▲                                                              │
│         │                          ┌──────────────────                │
│         │                         /│                                   │
│         │                        / │  Compute Bound                   │
│         │                       /  │  (Matrix multiply)               │
│         │                      /   │                                   │
│         │                     /    │                                   │
│         │                    /     │                                   │
│         │ Memory Bound     /      │                                   │
│         │ (LLM decode)    /       │                                   │
│         │                /        │                                   │
│         │               /         │                                   │
│         └──────────────┴──────────┴─────────────────────► FLOPs/Byte  │
│                                                                        │
│  LLM decode attention: ~10 FLOPs per byte loaded                      │
│  GPU can do: ~200 FLOPs per byte at memory bandwidth                  │
│                                                                        │
│  Result: GPU is waiting for memory 95% of the time!                   │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### Memory Access Patterns

```cpp
// BAD: Random access (cache thrashing)
for (int i = 0; i < N; i++) {
    int idx = random_index();
    value += data[idx];  // Memory access pattern is random
}

// GOOD: Sequential access (cache friendly)
for (int i = 0; i < N; i++) {
    value += data[i];  // Adjacent threads read adjacent memory
}

// BEST: Coalesced access (one memory transaction)
// Thread 0 reads data[0], Thread 1 reads data[1], ...
// GPU fetches all in one 128-byte transaction
```

---

## 2. The Tensor Class

### File: `cpp/common/tensor.h`

### Core Concept: Owned vs Borrowed Memory

```cpp
// The Tensor class has two modes:

// MODE 1: Tensor OWNS memory
// - Allocates memory in constructor
// - Frees memory in destructor
// - Like std::vector

rt::Tensor owned_tensor(
    {batch, seq_len, hidden},  // Shape
    rt::DeviceType::kGPU,      // Where
    nvinfer1::DataType::kHALF  // Type
);
// Memory allocated here!
// Memory freed when owned_tensor goes out of scope


// MODE 2: Tensor BORROWS memory
// - Points to existing memory
// - Does NOT free in destructor
// - Like a pointer/view

void* existing_memory = some_gpu_allocation;
rt::Tensor borrowed_tensor(
    existing_memory,           // External pointer
    {batch, seq_len, hidden},  // Shape
    rt::DeviceType::kGPU,      // Where
    nvinfer1::DataType::kHALF  // Type
);
// No allocation - just wrapping existing memory
// Memory NOT freed when borrowed_tensor goes out of scope
```

### Why Two Modes?

```
┌────────────────────────────────────────────────────────────────────────┐
│                     MEMORY OWNERSHIP PATTERNS                          │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  OWNED TENSORS (mOwnMemory = true)                                    │
│  ════════════════════════════════                                      │
│  Use when: Creating new allocations                                    │
│  Example: KV cache, execution context memory                          │
│                                                                        │
│      ┌─────────────────────────────────────┐                          │
│      │ Tensor A                            │                          │
│      │ mData ─────────► [GPU Memory Block] │                          │
│      │ mOwnMemory = true                   │                          │
│      │                                     │                          │
│      │ Destructor: cudaFree(mData)        │                          │
│      └─────────────────────────────────────┘                          │
│                                                                        │
│  BORROWED TENSORS (mOwnMemory = false)                                │
│  ════════════════════════════════════════                              │
│  Use when: Creating views of existing memory                          │
│  Example: Slicing KV cache for specific batch                         │
│                                                                        │
│      ┌─────────────────────────────────────┐                          │
│      │ Tensor B (view)                     │                          │
│      │ mData ─────────┐                    │                          │
│      │ mOwnMemory = false                  │                          │
│      └────────────────│────────────────────┘                          │
│                       │                                                │
│      ┌────────────────▼────────────────────┐                          │
│      │ Tensor A (owner)                    │                          │
│      │ mData ─────────► [GPU Memory Block] │                          │
│      │ mOwnMemory = true                   │                          │
│      └─────────────────────────────────────┘                          │
│                                                                        │
│      When B is destroyed: nothing happens (doesn't own memory)        │
│      When A is destroyed: cudaFree (owns memory)                      │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### Key Tensor Operations

```cpp
// File: cpp/common/tensor.h

// Get raw pointer for CUDA operations
void* ptr = tensor.rawPointer();
half* typed_ptr = tensor.dataPointer<half>();

// Get shape information
auto shape = tensor.getShape();      // Returns Coords (vector of dimensions)
int64_t total = tensor.getVolume();  // Total number of elements

// Get memory info
auto dtype = tensor.getDataType();   // kHALF, kFLOAT, kINT32, etc.
auto device = tensor.getDeviceType(); // kCPU or kGPU
size_t bytes = tensor.getSizeInBytes();
```

### Hands-on Exercise: Trace Tensor Lifecycle

```cpp
// Exercise: Read this code and predict memory behavior

void example() {
    // 1. Create owned tensor
    rt::Tensor a({1024, 768}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    // Q: How much memory was allocated?

    // 2. Create borrowed tensor pointing to same memory
    rt::Tensor b(a.rawPointer(), {1024, 768}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    // Q: How much NEW memory was allocated?

    // 3. Create another owned tensor
    rt::Tensor c({512, 768}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF);
    // Q: Total memory allocated so far?

}  // All tensors destroyed here
// Q: Which destructor calls cudaFree?

// Answers:
// 1. 1024 × 768 × 2 = 1.5 MB
// 2. 0 bytes (borrowed)
// 3. 1.5 MB + 0.75 MB = 2.25 MB total
// 4. Only 'a' and 'c' call cudaFree (they own memory)
```

---

## 3. KV Cache Architecture

### File: `cpp/runtime/linearKVCache.h`

### The "Linear" in Linear KV Cache

```
LINEAR (this system):
┌────────────────────────────────────────────────────────────────────────┐
│                    CONTIGUOUS MEMORY BLOCK                             │
│                                                                        │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐ │
│  │ L0  │ L0  │ L1  │ L1  │ L2  │ L2  │ ... │ L31 │ L31 │     │     │ │
│  │ K   │ V   │ K   │ V   │ K   │ V   │     │ K   │ V   │ ... │ ... │ │
│  │ S0  │ S0  │ S0  │ S0  │ S0  │ S0  │     │ S0  │ S0  │     │     │ │
│  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘ │
│                                                                        │
│  Advantages:                                                           │
│  ✓ Simple indexing: layer * stride + position                         │
│  ✓ Predictable memory access patterns                                 │
│  ✓ No pointer chasing                                                 │
│  ✓ Pre-allocated (no runtime allocation)                              │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘

PAGED (systems like vLLM):
┌────────────────────────────────────────────────────────────────────────┐
│                    SCATTERED MEMORY PAGES                              │
│                                                                        │
│  Page Table:                    Physical Pages:                        │
│  ┌─────────┐                    ┌───────┐  ┌───────┐  ┌───────┐       │
│  │ Page 0 ─┼───────────────────►│ Data  │  │ Data  │  │ Data  │       │
│  ├─────────┤                    └───────┘  └───┬───┘  └───┬───┘       │
│  │ Page 1 ─┼────────────────────────────────────┘          │          │
│  ├─────────┤                                               │          │
│  │ Page 2 ─┼───────────────────────────────────────────────┘          │
│  └─────────┘                                                          │
│                                                                        │
│  Advantages:                                                           │
│  ✓ Dynamic allocation (grow as needed)                                │
│  ✓ Better memory utilization with variable-length sequences           │
│                                                                        │
│  Disadvantages for edge:                                               │
│  ✗ Pointer indirection (slower)                                       │
│  ✗ Complex memory management                                          │
│  ✗ Page table overhead                                                │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### KV Cache Memory Layout

```
Shape: [num_layers, batch_size, 2, num_kv_heads, max_seq_len, head_dim]
                                 │
                                 └─ 2 = one for K, one for V

Example: 7B model, batch=1, max_seq=4096
  num_layers = 32
  batch_size = 1
  kv_factor = 2 (K and V)
  num_kv_heads = 8 (GQA: fewer KV heads than Q heads)
  max_seq_len = 4096
  head_dim = 128

Memory calculation:
  32 × 1 × 2 × 8 × 4096 × 128 × 2 bytes = 536 MB

Per-token:
  32 × 2 × 8 × 128 × 2 bytes = 131 KB per token
```

### Visualizing the Layout

```
┌────────────────────────────────────────────────────────────────────────┐
│                        KV CACHE MEMORY LAYOUT                          │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Layer 0:                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │ Batch 0:                                                         │  │
│  │ ┌───────────────────────────────────────────────────────────┐   │  │
│  │ │ K: [head0: ████████] [head1: ████████] ... [head7: ████] │   │  │
│  │ │    └─ 4096 positions × 128 dims each                      │   │  │
│  │ ├───────────────────────────────────────────────────────────┤   │  │
│  │ │ V: [head0: ████████] [head1: ████████] ... [head7: ████] │   │  │
│  │ └───────────────────────────────────────────────────────────┘   │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  Layer 1: (same structure)                                            │
│  ...                                                                   │
│  Layer 31: (same structure)                                           │
│                                                                        │
│  TOTAL: 32 layers × 2 (K,V) × 8 heads × 4096 pos × 128 dim × 2 bytes │
│       = 536 MB                                                        │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### Key KV Cache Operations

```cpp
// File: cpp/runtime/linearKVCache.h

class LinearKVCache {
public:
    // Reset cache for new batch
    void resetKVCache(cudaStream_t stream);

    // Get current sequence lengths per batch
    rt::Tensor& getKVCacheLengths();

    // After writing new tokens, update lengths
    void commitSequenceLength(int32_t increment, cudaStream_t stream);

    // Get raw pointer for attention kernels
    void* getKVCachePointer();

private:
    rt::Tensor mDeviceKVCache;        // The actual cache data
    rt::Tensor mDeviceKVCacheLengths; // Current length per sequence
    int32_t mMaxKVCacheCapacity;      // Maximum tokens
};
```

### Cache Growth During Generation

```
Token 0 (prefill input):
┌──────────────────────────────────────┐
│ Position 0: K₀, V₀                   │
│ Positions 1-4095: [empty]            │
└──────────────────────────────────────┘
mDeviceKVCacheLengths = [1]

After prefill (100 input tokens):
┌──────────────────────────────────────┐
│ Position 0-99: K₀..K₉₉, V₀..V₉₉     │ ◄── Filled by prefill
│ Positions 100-4095: [empty]          │
└──────────────────────────────────────┘
mDeviceKVCacheLengths = [100]

After 50 decode steps:
┌──────────────────────────────────────┐
│ Position 0-99: [original input]      │
│ Position 100-149: [generated]        │ ◄── Added during decode
│ Positions 150-4095: [empty]          │
└──────────────────────────────────────┘
mDeviceKVCacheLengths = [150]
```

---

## 4. Memory Pre-allocation Strategy

### The Problem with Dynamic Allocation

```
Dynamic allocation (BAD for inference):
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Token 1:   cudaMalloc ──────────────────────────► 1ms             │
│             compute    ─────► 30ms                                 │
│                                                                     │
│  Token 2:   cudaMalloc ──────────────────────────► 1ms             │
│             compute    ─────► 30ms                                 │
│                                                                     │
│  Token 100: cudaMalloc ──────────────────────────► 1ms             │
│             compute    ─────► 30ms                                 │
│                                                                     │
│  Total time: 100 × (1 + 30) = 3100ms                               │
│  Wasted on malloc: 100ms (3.2%)                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Pre-allocation (GOOD for inference):
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Startup:   cudaMalloc for ALL memory ──────────► 10ms (once!)     │
│                                                                     │
│  Token 1:   compute    ─────► 30ms                                 │
│  Token 2:   compute    ─────► 30ms                                 │
│  Token 100: compute    ─────► 30ms                                 │
│                                                                     │
│  Total time: 10 + 100 × 30 = 3010ms                                │
│  Wasted on malloc: 10ms (0.3%)                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### What Gets Pre-allocated

```cpp
// File: cpp/runtime/llmEngineRunner.cpp (constructor)

// 1. Execution context memory (shared between prefill and generation)
mExecContextMemory = rt::Tensor(
    {execContextMemoryInBytes},
    rt::DeviceType::kGPU,
    nvinfer1::DataType::kUINT8
);

// 2. KV cache (for storing K, V across all layers)
mKVCache.initialize(
    numDecoderLayers,
    maxBatchSize,
    numKVHeads,
    maxKVCacheCapacity,
    headDim,
    stream
);

// 3. Rope cos/sin cache (positional encodings)
mPosEncCosSinCache = rt::Tensor(...);

// 4. Working tensors (reused each step)
mSequenceContextLengths = rt::Tensor(...);
mSelectTokenIndices = rt::Tensor(...);
```

### Memory Budget Calculation

```python
# Python pseudo-code for memory planning

def calculate_memory_budget(model_config, runtime_config):
    """Calculate total GPU memory needed."""

    # Model weights (loaded separately)
    weights_memory = model_config.num_params * bytes_per_param
    # 7B params × 1 byte (FP8) = 7GB

    # KV cache
    kv_cache_memory = (
        model_config.num_layers *
        runtime_config.max_batch_size *
        2 *  # K and V
        model_config.num_kv_heads *
        runtime_config.max_kv_capacity *
        model_config.head_dim *
        2  # FP16
    )
    # 32 × 1 × 2 × 8 × 4096 × 128 × 2 = 536MB

    # Activations (during prefill, largest tensor)
    activations_memory = (
        runtime_config.max_batch_size *
        runtime_config.max_input_len *
        model_config.hidden_size *
        2  # FP16
    )
    # 1 × 2048 × 4096 × 2 = 16MB

    # Workspace (for intermediate computations)
    workspace_memory = 512 * 1024 * 1024  # 512MB typically

    total = weights_memory + kv_cache_memory + activations_memory + workspace_memory
    return total

# Example for 7B model on Jetson Orin 32GB:
# Weights: 7GB (FP8)
# KV Cache: 0.5GB (4K context)
# Activations: 0.5GB
# Workspace: 0.5GB
# Total: ~8.5GB
# Headroom: 32 - 8.5 = 23.5GB available
```

### Hands-on Exercise: Memory Profiling

```bash
# Profile memory usage during inference
./build/examples/llm/llm_inference \
    --engineDir ./engines/model \
    --inputFile input.json \
    --outputFile output.json \
    --dumpProfile \
    --profileOutputFile profile.json

# In another terminal, watch GPU memory:
watch -n 0.5 nvidia-smi

# Note:
# 1. Memory usage at startup (all allocations)
# 2. Memory usage during prefill (peak)
# 3. Memory usage during decode (should be stable)
```

---

## Feynman Self-Test

- [ ] **Why is LLM inference memory-bound?**
  > The GPU can compute faster than memory can deliver data. Most time is spent waiting for memory transfers.

- [ ] **What's the difference between owned and borrowed tensors?**
  > Owned tensors allocate and free memory. Borrowed tensors just point to existing memory (views).

- [ ] **Why use Linear KV cache instead of Paged?**
  > Linear is simpler, has predictable access patterns, no pointer chasing - better for edge where simplicity = speed.

- [ ] **How much memory does the KV cache use for 7B model at 4K context?**
  > ~500MB (32 layers × 8 heads × 4096 positions × 128 dims × 2 for K,V × 2 bytes)

- [ ] **Why pre-allocate everything?**
  > cudaMalloc is slow (~1ms). Pre-allocating once at startup avoids latency spikes during generation.

## If You're Stuck

### "Memory calculations don't add up"
Double-check your units:
- FP16 = 2 bytes per element
- FP8 = 1 byte per element
- 1MB = 1,000,000 bytes (not 1,024,000)

### "Can't find memory allocation in code"
```bash
grep -r "cudaMalloc" cpp/
grep -r "Tensor(" cpp/runtime/
```

### "Why is my memory usage higher than calculated?"
- TensorRT reserves workspace memory
- CUDA context overhead (~500MB)
- Memory fragmentation

---

## What's Next?

You now understand:
- ✅ GPU memory hierarchy and why bandwidth matters
- ✅ Tensor ownership model
- ✅ KV cache architecture and layout
- ✅ Pre-allocation strategy

**Next**: [04 Python Pipeline](04_python_pipeline.md) - Deep dive into quantization, ONNX export, and model surgery.

---

*← [02 Architecture](02_architecture.md) | [04 Python Pipeline →](04_python_pipeline.md)*
