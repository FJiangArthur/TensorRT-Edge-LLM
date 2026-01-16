# Level 4a: Optimization Techniques

**Reading Time: 2 hours | Hands-on Exercises: 6**

You've learned how inference works. Now let's learn how to make it *fast*. Every millisecond matters on edge devices - this level covers every optimization technique that makes TensorRT-Edge-LLM competitive.

## The Core Insight First

1. **LLM inference is memory-bound, not compute-bound** - GPU waits for data, not math
2. **Latency has two enemies: allocation and launch overhead** - we eliminate both
3. **The key insight: do everything once, reuse forever**

## Key Numbers to Memorize

| Optimization | Typical Speedup | When It Helps Most |
|--------------|-----------------|-------------------|
| **CUDA Graphs** | 5-15% latency | Decode phase (many small launches) |
| **Memory Pre-allocation** | 5-15% latency | All phases (avoids cudaMalloc) |
| **Flash Attention** | 2-4x throughput | Long sequences (>512 tokens) |
| **INT4 Quantization** | 2-4x throughput | Memory-bound workloads |
| **Vocab Reduction** | Up to 4x on LM head | Domain-specific models |
| **KV Cache Reuse** | 1.5-2x prefill | Shared system prompts |

## Table of Contents

1. [CUDA Graphs: Record Once, Replay Forever](#1-cuda-graphs-record-once-replay-forever)
2. [Memory Pre-allocation: Zero Runtime Mallocs](#2-memory-pre-allocation-zero-runtime-mallocs)
3. [Flash Attention: The Algorithmic Win](#3-flash-attention-the-algorithmic-win)
4. [Quantization: Trading Precision for Speed](#4-quantization-trading-precision-for-speed)
5. [Vocabulary Reduction: Smaller Output Layer](#5-vocabulary-reduction-smaller-output-layer)
6. [KV Cache Optimizations](#6-kv-cache-optimizations)

---

## 1. CUDA Graphs: Record Once, Replay Forever

### The Problem

Every time you run a CUDA kernel, the CPU does work:
```
CPU: Prepare arguments → Launch kernel → Wait for GPU → Prepare next → Launch next → ...
```

On Jetson, CPU is slower than on desktop. This overhead adds up during decode:
- 100 tokens generated = 100 kernel launches
- Each launch ~0.5ms CPU overhead
- Total: 50ms wasted just on launches!

### The Solution: CUDA Graphs

**Mental Model**: Think of it like a recorded macro in Excel.

```
First run:  CPU records all the GPU commands into a "tape"
Later runs: CPU just says "play the tape" - GPU does everything

┌─────────────────────────────────────────────────────────────────┐
│                     WITHOUT CUDA GRAPHS                         │
├─────────────────────────────────────────────────────────────────┤
│  CPU: [prep][launch]───[prep][launch]───[prep][launch]───...   │
│  GPU:       [kernel]──────────[kernel]──────────[kernel]───... │
│                   ↑               ↑               ↑             │
│               idle time       idle time       idle time         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      WITH CUDA GRAPHS                           │
├─────────────────────────────────────────────────────────────────┤
│  CPU: [graph launch]────────────────────────────────────────... │
│  GPU: [kernel][kernel][kernel][kernel][kernel][kernel]───────...│
│        ↑                                                        │
│   GPU runs everything back-to-back, no CPU involvement!        │
└─────────────────────────────────────────────────────────────────┘
```

### How It's Implemented

**File**: `cpp/runtime/llmEngineRunner.cpp` (lines 1268-1381)

```cpp
// Step 1: Calculate a hash to identify this exact configuration
size_t graphHash = hashDecodingInput(inputIds, outputLogits, mActiveLoraWeightsName);

// Step 2: Check if we already captured a graph for this config
if (mCudaGraphs.find(graphHash) == mCudaGraphs.end()) {
    // First time - capture the graph

    // Run once to let TensorRT "warm up" its shape machinery
    mGenerationExecutionContext->enqueueV3(stream);
    cudaStreamSynchronize(stream);

    // Now capture
    cudaGraph_t graph;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    mGenerationExecutionContext->enqueueV3(stream);  // This gets recorded!
    cudaStreamEndCapture(stream, &graph);

    // Instantiate for replay
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, 0);

    // Save for later
    mCudaGraphs[graphHash] = {graph, graphExec};
}

// Step 3: Replay the captured graph
cudaGraphExec_t graphExec = mCudaGraphs[graphHash].second;
cudaGraphLaunch(graphExec, stream);  // One call launches everything!
```

### Why We Need Multiple Graphs

The hash includes:
- **Batch size**: Different batch = different tensor shapes
- **Tensor addresses**: Graph records exact memory locations
- **LoRA adapter name**: Different adapters = different weights

```cpp
// From llmEngineRunner.cpp - the hash function
size_t LLMEngineRunner::hashDecodingInput(
    rt::Tensor const& inputIds,
    rt::Tensor const& outputLogits,
    std::string const& loraWeightsName)
{
    size_t hash = 0;
    combineHash(hash, inputIds.getShape()[0]);      // batch size
    combineHash(hash, inputIds.rawPointer());       // input address
    combineHash(hash, outputLogits.rawPointer());   // output address
    combineHash(hash, loraWeightsName);             // LoRA adapter
    return hash;
}
```

### Exercise 6.1: Trace CUDA Graph Capture

**Task**: Find and understand the complete capture flow.

```bash
# Search for graph capture in the codebase
grep -n "cudaStreamBeginCapture" cpp/runtime/*.cpp
grep -n "cudaGraphLaunch" cpp/runtime/*.cpp
```

**Questions to answer** (check `cpp/runtime/llmEngineRunner.cpp`):

1. What happens if you change batch size mid-inference?
   - Hint: Look at `hashDecodingInput()`

2. Why is there a `cudaStreamSynchronize()` before capture?
   - Hint: TensorRT needs to finalize internal state

3. What's in `mBaseTreeDecodingCudaGraphs` vs `mCudaGraphs`?
   - Hint: One is for vanilla decode, one is for EAGLE

**Your answers**:
```
1. _____________________________________________

2. _____________________________________________

3. _____________________________________________
```

---

## 2. Memory Pre-allocation: Zero Runtime Mallocs

### The Problem

`cudaMalloc()` is expensive:
- ~1ms per allocation
- Fragments GPU memory over time
- Can fail unexpectedly if memory is fragmented

For 100-token generation with per-token allocation: 100ms wasted!

### The Solution: Allocate Everything Upfront

**Mental Model**: Like setting up a assembly line before production starts.

```
┌────────────────────────────────────────────────────────────────┐
│                AT ENGINE LOAD TIME (ONCE)                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────┐  │
│  │  KV Cache       │  │ Execution Ctx   │  │ RoPE Cache    │  │
│  │  (14GB for 7B)  │  │ Memory          │  │ (cos/sin)     │  │
│  └─────────────────┘  └─────────────────┘  └───────────────┘  │
│                                                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────┐  │
│  │  Input Tensors  │  │ Output Tensors  │  │ Scratch Space │  │
│  │  (max batch)    │  │ (max vocab)     │  │               │  │
│  └─────────────────┘  └─────────────────┘  └───────────────┘  │
│                                                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                AT INFERENCE TIME (EVERY REQUEST)               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  No allocations! Just fill pre-allocated buffers with data.   │
│                                                                │
│  cudaMalloc: 0 times                                          │
│  cudaMemcpy: as needed (data only)                            │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Key Pre-allocations

**1. KV Cache** (`cpp/runtime/linearKVCache.cpp:34-47`)
```cpp
LinearKVCache::LinearKVCache(CacheConfig const& config, cudaStream_t stream)
{
    // Calculate total cache size
    int64_t kvCacheVolume = mConfig.numDecoderLayers
                          * mConfig.maxBatchSize
                          * 2  // K and V
                          * mConfig.numKVHeads
                          * mConfig.maxSequenceLength
                          * mConfig.headDim;

    // One allocation for entire cache
    CUDA_CHECK(cudaMalloc(&mDeviceKVCache, kvCacheVolume * sizeof(KVCacheType)));

    // Memory layout: [layers][batch][K/V][heads][seq][dim]
}
```

**2. Execution Context Memory** (`cpp/runtime/llmEngineRunner.cpp:154-167`)
```cpp
// TensorRT needs scratch space for intermediate activations
int64_t execContextMemoryInBytes = mEngine->getDeviceMemorySizeV2();
mExecContextMemory = rt::Tensor({execContextMemoryInBytes}, ...);

// Share between prefill and generation (they never run simultaneously)
mPrefillExecutionContext->setDeviceMemoryV2(
    mExecContextMemory.rawPointer(), execContextMemoryInBytes);
mGenerationExecutionContext->setDeviceMemoryV2(
    mExecContextMemory.rawPointer(), execContextMemoryInBytes);
```

**3. RoPE Cache** (`cpp/runtime/llmEngineRunner.cpp:190-241`)
```cpp
// Pre-compute cos/sin values for all possible positions
// Much faster than computing on-the-fly during attention

// For each position 0..maxSeqLen:
//   ropeCache[pos] = {cos(pos * theta), sin(pos * theta)}
```

### Exercise 6.2: Calculate Memory Requirements

**Task**: Calculate the memory footprint for a 7B model.

Given:
- 32 decoder layers
- 8 KV heads
- 128 head dimension
- 4096 max sequence length
- Batch size 4
- FP16 precision (2 bytes)

**KV Cache size** = layers × batch × 2 × heads × seq × dim × bytes

```
= 32 × 4 × 2 × 8 × 4096 × 128 × 2
= ____________ bytes
= ____________ GB
```

**Questions**:
1. If you reduce `maxSequenceLength` to 2048, how much memory do you save?
2. Why do we multiply by 2 (the K/V factor)?
3. What happens if you try to process a 5000-token sequence with maxSeqLen=4096?

---

## 3. Flash Attention: The Algorithmic Win

### The Problem with Standard Attention

Standard attention computes:
```
Attention(Q, K, V) = softmax(Q × K^T / sqrt(d)) × V
```

The problem? `Q × K^T` creates a huge matrix:
- Sequence length 4096 → 4096 × 4096 = 16 million elements
- Per head, per batch, per layer
- Must read/write this matrix to GPU memory (slow!)

### Flash Attention: Block-Wise Computation

**Mental Model**: Instead of computing the entire attention matrix, compute it in small blocks and keep running statistics.

```
┌─────────────────────────────────────────────────────────────────┐
│              STANDARD ATTENTION (Memory Heavy)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: S = Q × K^T              ← Write 16M elements to HBM  │
│  Step 2: P = softmax(S)           ← Read/Write 16M elements    │
│  Step 3: O = P × V                ← Read 16M elements          │
│                                                                 │
│  Total HBM access: 48M elements × 2 bytes = 96MB per head!    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              FLASH ATTENTION (Memory Efficient)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For each block of Q (fits in SRAM):                           │
│    For each block of K,V (fits in SRAM):                       │
│      - Compute partial attention in SRAM (fast!)               │
│      - Update running softmax statistics                        │
│      - Accumulate partial output                                │
│    End                                                          │
│  End                                                            │
│                                                                 │
│  Never materialize full attention matrix!                       │
│  HBM access: O(N) instead of O(N²)                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Online Softmax: The Key Trick

Standard softmax needs to see all values first:
```
softmax(x) = exp(x) / sum(exp(x))  ← Need sum over ALL values!
```

Online softmax updates incrementally:
```cpp
// Process block by block
for each block:
    local_max = max(block)
    if local_max > global_max:
        // Rescale previous sum
        sum = sum * exp(global_max - local_max)
        global_max = local_max

    sum += sum(exp(block - global_max))
```

### Implementation in TensorRT-Edge-LLM

**File**: `cpp/kernels/contextAttentionKernels/contextFMHARunner.cpp`

The project uses NVIDIA's Flash Multi-Head Attention v2 kernels:
```
contextAttentionKernels/cubin/
├── fmha_v2_flash_attention_fp16_64_128_S_q_kv_128_sm80.cubin   # Ampere
├── fmha_v2_flash_attention_fp16_64_128_S_q_kv_128_sm86.cubin   # GA102
├── fmha_v2_flash_attention_fp16_64_128_S_q_kv_128_sm87.cubin   # Orin
└── ... (different SM versions, head sizes, precisions)
```

Different kernels for:
- **SM version**: 80 (A100), 86 (RTX 3090), 87 (Orin), 89 (Ada), 100+ (Blackwell)
- **Head size**: 32, 64, 128 dimensions
- **Format**: Separate Q+KV or combined QKV

### Exercise 6.3: Memory Savings Calculation

**Task**: Calculate memory savings from Flash Attention.

Given:
- Sequence length: 4096
- Heads: 32
- Batch: 4
- FP16 precision

**Standard Attention memory** (just the attention matrix):
```
= batch × heads × seq × seq × 2 bytes
= 4 × 32 × 4096 × 4096 × 2
= ____________ bytes
= ____________ GB
```

**Flash Attention memory** (block size 128):
```
= batch × heads × 2 × block_size × seq × 2 bytes
= 4 × 32 × 2 × 128 × 4096 × 2
= ____________ bytes
= ____________ MB
```

**Savings ratio**: ____________

---

## 4. Quantization: Trading Precision for Speed

### Why Quantization Helps

LLM inference is memory-bound. The GPU can compute faster than it can load weights.

```
┌─────────────────────────────────────────────────────────────────┐
│                    MEMORY BANDWIDTH BOTTLENECK                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Jetson Orin memory bandwidth: ~200 GB/s                       │
│  7B model weights (FP16): 14 GB                                │
│                                                                 │
│  Time to load all weights once: 14 GB / 200 GB/s = 70ms       │
│                                                                 │
│  With INT4 quantization:                                       │
│  7B model weights (INT4): 3.5 GB                               │
│  Time to load: 3.5 GB / 200 GB/s = 17.5ms                     │
│                                                                 │
│  4x speedup just from reduced memory traffic!                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Quantization Schemes

| Scheme | Bits | Size Reduction | Quality Impact |
|--------|------|----------------|----------------|
| **FP16** | 16 | 1x (baseline) | None |
| **FP8** | 8 | 2x | Minimal |
| **INT8** | 8 | 2x | Low |
| **INT4 AWQ** | 4 | 4x | Moderate |
| **NVFP4** | 4 | 4x | Low (newer) |

### INT4 Group-Wise Quantization

**File**: `tensorrt_edgellm/llm_models/layers/int4_gemm_plugin.py`

Group-wise means: different scale per group of weights (typically 128 weights per group).

```python
# Original weight: [K, N] in FP16
# Quantized:
#   - qweight: [K, N/2] in INT8 (packed: 2 INT4 per byte)
#   - scales:  [K/group_size, N] in FP16
#   - zeros:   [K/group_size, N] in INT8

# Dequantization formula:
# weight[k, n] = (qweight[k, n] - zeros[k//gs, n]) * scales[k//gs, n]
```

### The GEMM Kernel

**File**: `cpp/kernels/int4GroupwiseGemmKernels/int4GroupwiseGemm.h`

```cpp
// For small batches (M=1-4), use GEMV (optimized for vectors)
void gemv_forward_cuda_new(
    half* in_feats,           // [M, K] input activations
    int8_t* kernel,           // [N/2, K] packed INT4 weights
    half* scaling_factors,    // [K/gs, N] per-group scales
    half* out_feats,          // [M, N] output
    int m, int n, int k, int group_size, cudaStream_t stream);

// For larger batches, use GEMM
void gemm_forward_cuda_new(
    half* in_feats,
    int8_t* kernel,
    half* scaling_factors,
    half* out_feats,
    int m, int n, int k, int group_size, cudaStream_t stream);
```

Key optimization: **Fused dequantization**
- Don't convert INT4→FP16 separately (would need write/read)
- Dequantize on-the-fly during GEMM computation
- Weight stays in INT4 format in memory

### Exercise 6.4: Quantization Trade-offs

**Task**: Analyze when to use each quantization scheme.

Fill in the table based on your understanding:

| Scenario | Recommended Scheme | Why |
|----------|-------------------|-----|
| Jetson Orin 32GB, 7B model, quality-critical | _______ | _______ |
| Jetson Orin 32GB, 7B model, speed-critical | _______ | _______ |
| Jetson Orin 16GB, 7B model | _______ | _______ |
| x86 with A100, latency-critical | _______ | _______ |

**Bonus**: Find the quantization command in the codebase:
```bash
grep -r "quantize" tensorrt_edgellm/scripts/ --include="*.py"
```

---

## 5. Vocabulary Reduction: Smaller Output Layer

### The Problem

The final layer (LM head) converts hidden states to vocabulary probabilities:
- Hidden dim: 4096
- Vocab size: 32000 (typical)
- Weight matrix: 4096 × 32000 = 131M parameters
- That's a big GEMM for every generated token!

### The Solution: Use Fewer Vocabulary Entries

**Mental Model**: If you're only doing code generation, you don't need emojis in your vocabulary.

```
┌─────────────────────────────────────────────────────────────────┐
│                   VOCABULARY REDUCTION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Full vocabulary: 32,000 tokens                                │
│  ├── Common English words                                      │
│  ├── Code tokens                                               │
│  ├── Emojis                           ← Not needed for code!  │
│  ├── Foreign language characters      ← Not needed for code!  │
│  └── Rare symbols                     ← Not needed for code!  │
│                                                                 │
│  Reduced vocabulary: 8,000 tokens                              │
│  ├── Common English words (filtered)                           │
│  └── Code tokens (all)                                         │
│                                                                 │
│  LM Head GEMM: 4x smaller! (32k → 8k output dimension)        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

**File**: `tensorrt_edgellm/vocab_reduction/vocab_reduction.py`

Two reduction methods:
1. **Frequency-based**: Keep most common tokens in your dataset
2. **Task-aware**: Analyze your specific task's token distribution

```python
def reduce_vocab_size(
    tokenizer,
    config,
    dataset,
    reduced_vocab_size: int,  # How many tokens to keep
    method: str = 'frequency'
) -> torch.Tensor:
    """
    Returns vocab_map: mapping from reduced IDs to original IDs

    Example:
      vocab_map[0] = 256   # Reduced token 0 maps to original token 256
      vocab_map[1] = 13    # Reduced token 1 maps to original token 13
      ...
    """
```

**File**: `tensorrt_edgellm/llm_models/layers/reduced_lm_head.py`

```python
def reduce_lm_head(lm_head: nn.Linear, reduced_vocab_size: int,
                   vocab_map: torch.Tensor) -> nn.Linear:
    """
    Create a smaller LM head using only needed vocabulary entries.

    Original: weight shape (32000, 4096)
    Reduced:  weight shape (8000, 4096) using vocab_map indices
    """
    # Select only the rows we need
    new_weight = lm_head.weight[vocab_map]  # (8000, 4096)

    # Create new smaller layer
    new_lm_head = nn.Linear(4096, 8000, bias=False)
    new_lm_head.weight = nn.Parameter(new_weight)

    return new_lm_head
```

### Exercise 6.5: Vocab Reduction Analysis

**Task**: Determine if vocab reduction is right for your use case.

Consider these scenarios:

| Use Case | Can Use Vocab Reduction? | Expected Savings |
|----------|-------------------------|------------------|
| General chatbot | _______ | _______ |
| Code completion (Python only) | _______ | _______ |
| Medical report summarization | _______ | _______ |
| Multi-language translation | _______ | _______ |

**Question**: What happens if the model tries to generate a token not in the reduced vocabulary?

Hint: Check the sampling code and `vocab_map` usage.

---

## 6. KV Cache Optimizations

### Linear vs Paged KV Cache

TensorRT-Edge-LLM uses **Linear KV Cache** (not paged like vLLM).

```
┌─────────────────────────────────────────────────────────────────┐
│                    LINEAR KV CACHE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Memory layout: [layer][batch][K/V][head][seq][dim]            │
│                                                                 │
│  ┌────────────────────────────────────────────────┐            │
│  │ Seq0: [K0][K1][K2][K3][--][--][--][--]        │ ← contiguous│
│  │ Seq1: [K0][K1][K2][--][--][--][--][--]        │             │
│  │ Seq2: [K0][K1][--][--][--][--][--][--]        │             │
│  └────────────────────────────────────────────────┘            │
│                                                                 │
│  Pros: Simple, predictable, fast sequential access             │
│  Cons: Wastes memory for short sequences                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    PAGED KV CACHE (vLLM style)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Memory layout: scattered pages with page table                 │
│                                                                 │
│  ┌─────────────────────────────────────────────────┐           │
│  │ Page Table: [Seq0 → pages 2,5,7] [Seq1 → 0,3]  │           │
│  │                                                 │           │
│  │ Page 0: [Seq1 data]                            │           │
│  │ Page 2: [Seq0 data]     ← non-contiguous       │           │
│  │ Page 3: [Seq1 data]                            │           │
│  │ Page 5: [Seq0 data]                            │           │
│  └─────────────────────────────────────────────────┘           │
│                                                                 │
│  Pros: Memory efficient for variable-length sequences          │
│  Cons: Complex, page table overhead, scattered access          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Why Linear for Edge?**
- Single-user inference (no dynamic batching)
- Simpler = faster for predictable workloads
- No page table management overhead

### System Prompt Caching

**File**: `cpp/runtime/llmInferenceRuntime.h` (lines 35-42)

Common pattern: Same system prompt for every request.

```cpp
struct SystemPromptKVCache
{
    std::string systemPrompt;                      // The original text
    std::vector<tokenizer::Rank> tokenizedPrompt; // Token IDs
    rt::Tensor kvCacheContent;                    // PRE-COMPUTED KV!
};
```

**How it works**:
```
Request 1: System prompt (500 tokens) + User message (50 tokens)
  → Prefill: 550 tokens
  → Save system prompt KV to cache

Request 2: Same system prompt + Different user message
  → Load system prompt KV from cache (skip 500 tokens!)
  → Prefill: Only 50 new tokens

Speedup: 11x faster prefill!
```

### Cache Key Design

```cpp
// Hash function for system prompt cache lookup
size_t systemPromptHash = hashCombine(
    std::hash<std::string>{}(systemPrompt),
    std::hash<std::string>{}(loraAdapterName)  // Different LoRA = different cache!
);
```

### Exercise 6.6: KV Cache Sizing

**Task**: Calculate optimal KV cache configuration.

Your setup:
- Jetson Orin with 32GB VRAM
- 7B model (14GB weights in FP16, 3.5GB in INT4)
- Want batch size 4
- Target max context: 4096 tokens

**Memory budget for KV cache**:
```
Total VRAM:        32 GB
Model weights:     ____ GB (FP16 or INT4?)
CUDA overhead:     ~2 GB
Execution context: ~2 GB
Available for KV:  ____ GB
```

**KV cache size calculation** (FP16):
```
32 layers × 4 batch × 2 (K+V) × 8 heads × 4096 seq × 128 dim × 2 bytes
= ____________ bytes = ____________ GB
```

**Does it fit?** ____

**If not, what can you reduce?**
1. Batch size: ____
2. Max sequence length: ____
3. Use INT8 KV cache: ____

---

## Putting It All Together

### The Optimization Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    OPTIMIZATION LAYERS                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 5: Application Level                                    │
│  ┌─────────────────────────────────────────────────────┐       │
│  │ System Prompt Caching │ Vocab Reduction            │       │
│  └─────────────────────────────────────────────────────┘       │
│                              │                                  │
│  Layer 4: Execution Level                                      │
│  ┌─────────────────────────────────────────────────────┐       │
│  │ CUDA Graphs │ Memory Pre-allocation                 │       │
│  └─────────────────────────────────────────────────────┘       │
│                              │                                  │
│  Layer 3: Algorithm Level                                      │
│  ┌─────────────────────────────────────────────────────┐       │
│  │ Flash Attention │ Online Softmax                    │       │
│  └─────────────────────────────────────────────────────┘       │
│                              │                                  │
│  Layer 2: Arithmetic Level                                     │
│  ┌─────────────────────────────────────────────────────┐       │
│  │ INT4/FP8 Quantization │ Fused Kernels              │       │
│  └─────────────────────────────────────────────────────┘       │
│                              │                                  │
│  Layer 1: Hardware Level                                       │
│  ┌─────────────────────────────────────────────────────┐       │
│  │ TensorRT Engine Optimization │ Tensor Core Usage    │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Which Optimizations Apply When?

| Optimization | Prefill Phase | Decode Phase |
|--------------|--------------|--------------|
| CUDA Graphs | No (variable shapes) | Yes (fixed shapes) |
| Flash Attention | Yes (biggest win) | Yes (smaller win) |
| Quantization | Yes | Yes |
| Vocab Reduction | No (no LM head) | Yes |
| KV Cache Reuse | Yes (skips prefill) | No |
| Memory Pre-alloc | Yes | Yes |

---

## Feynman Self-Test

After completing this level, you should be able to answer:

- [ ] **Why does CUDA graph capture need multiple graphs per model?**
  > Because tensor addresses and batch sizes are baked into the graph

- [ ] **What makes LLM inference memory-bound?**
  > GPU can compute faster than it can load weights from memory

- [ ] **How does Flash Attention avoid O(N²) memory?**
  > Block-wise computation with online softmax - never materializes full attention matrix

- [ ] **When is vocabulary reduction a bad idea?**
  > When the model needs to output tokens not in the reduced set

- [ ] **Why use Linear KV cache instead of Paged on edge?**
  > Simpler, faster for single-user inference, no page table overhead

---

## Quick Reference

```bash
# Enable CUDA graphs (typically enabled by default)
# Check in config.json: "useCudaGraph": true

# Quantization options during export
tensorrt-edgellm-quantize-llm --quantization fp8      # Balanced
tensorrt-edgellm-quantize-llm --quantization int4_awq # Max compression

# Vocabulary reduction during export
tensorrt-edgellm-export-llm --reduced_vocab_size 8000

# Build with specific KV cache size
./llm_build --maxKVCacheCapacity 2048  # Reduce for memory-constrained devices
```

---

## What's Next?

You've mastered the optimization techniques! You can now:
- ✅ Explain every optimization in the stack
- ✅ Calculate memory requirements for any configuration
- ✅ Choose the right trade-offs for your use case

**Next Level**: [07 Advanced Features](07_advanced_features.md) - EAGLE speculative decoding, LoRA switching, and VLM support.

---

*← [05 C++ Runtime](05_cpp_runtime.md) | [07 Advanced Features →](07_advanced_features.md)*
