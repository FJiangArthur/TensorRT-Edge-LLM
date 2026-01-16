# Level 3b: C++ Runtime Internals Deep Dive

**Reading Time: 2 hours**

Imagine you're an air traffic controller. Planes (tokens) need to land (be generated) one at a time on a runway (GPU). Your job is to coordinate departures, arrivals, and refueling (memory management) so nothing crashes and planes move as fast as possible. The C++ runtime is that air traffic controller - it orchestrates every GPU operation to generate tokens efficiently. This tutorial teaches you to read and understand that control system.

## The Core Insight First

1. **The runtime is a state machine** - It transitions between prefill, decode, and sampling states
2. **Everything is pre-bound** - Tensor addresses are set once, then reused
3. **The decode loop is a hot path** - Every microsecond saved here matters

## Key Numbers to Memorize

| Component | Latency | Where to Optimize |
|-----------|---------|-------------------|
| **TensorRT enqueue** | ~100μs | Engine execution overhead |
| **CUDA kernel launch** | ~5μs | Per kernel, adds up |
| **Memory copy H→D** | ~1ms/MB | Avoid during inference |
| **cudaStreamSynchronize** | ~10μs | Only when necessary |
| **Attention kernel** | 10-30ms | The bottleneck |

## Table of Contents

1. [Runtime Architecture](#1-runtime-architecture) (30 min)
2. [The Decode Loop](#2-the-decode-loop) (30 min)
3. [Attention Kernels](#3-attention-kernels) (30 min)
4. [CUDA Graphs](#4-cuda-graphs) (30 min)

---

## 1. Runtime Architecture

### Class Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RUNTIME CLASS HIERARCHY                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LLMInferenceRuntime  (High-level API)                                      │
│  ════════════════════════════════════                                       │
│  File: cpp/runtime/llmInferenceRuntime.h                                    │
│  Role: Handle requests, manage tokenization, orchestrate generation         │
│         │                                                                   │
│         │ owns                                                              │
│         ▼                                                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  LLMEngineRunner  (Engine execution)                                   │ │
│  │  ═════════════════════════════════════                                │ │
│  │  File: cpp/runtime/llmEngineRunner.h                                   │ │
│  │  Role: Execute TensorRT engine, manage KV cache                       │ │
│  │         │                                                              │ │
│  │         │ owns                                                         │ │
│  │         ▼                                                              │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │  LinearKVCache                                                   │  │ │
│  │  │  ═════════════════                                               │  │ │
│  │  │  File: cpp/runtime/linearKVCache.h                               │  │ │
│  │  │  Role: Manage KV cache memory                                    │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                        │ │
│  │         │ uses                                                         │ │
│  │         ▼                                                              │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │  TensorRT Engine (nvinfer1::ICudaEngine)                        │  │ │
│  │  │  ═════════════════════════════════════════                      │  │ │
│  │  │  Compiled model, executes forward pass                          │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│         │ uses                                                              │
│         ▼                                                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  Tokenizer                                                             │ │
│  │  ══════════                                                            │ │
│  │  File: cpp/tokenizer/tokenizer.cpp                                     │ │
│  │  Role: Text ↔ token IDs                                               │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│         │ uses                                                              │
│         ▼                                                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  Sampler                                                               │ │
│  │  ═════════                                                             │ │
│  │  File: cpp/sampler/sampling.h                                          │ │
│  │  Role: Logits → next token ID                                         │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Request Flow Through Runtime

```cpp
// File: cpp/runtime/llmInferenceRuntime.cpp

bool LLMInferenceRuntime::handleRequest(
    LLMGenerationRequest const& request,
    LLMGenerationResponse& response,
    cudaStream_t stream)
{
    // PHASE 1: TOKENIZATION
    // ═════════════════════
    std::vector<std::vector<int32_t>> batchedInputIds;
    for (auto const& req : request.requests) {
        // Apply chat template
        std::string formatted = applyChatTemplate(req.messages);
        // Tokenize
        std::vector<int32_t> tokens = mTokenizer->encode(formatted);
        batchedInputIds.push_back(tokens);
    }

    // PHASE 2: PREFILL
    // ════════════════
    // Process all input tokens, fill KV cache
    runPrefill(batchedInputIds, stream);

    // PHASE 3: DECODE LOOP
    // ════════════════════
    // Generate tokens one at a time
    std::vector<std::vector<int32_t>> outputIds(request.requests.size());
    for (int step = 0; step < request.maxGenerateLength; ++step) {
        // Run one decode step
        runDecodeStep(stream);

        // Sample next tokens
        std::vector<int32_t> nextTokens = sampleTokens(request, stream);

        // Check for completion (EOS token)
        bool allDone = true;
        for (size_t i = 0; i < nextTokens.size(); ++i) {
            outputIds[i].push_back(nextTokens[i]);
            if (nextTokens[i] != mTokenizer->eosTokenId()) {
                allDone = false;
            }
        }
        if (allDone) break;
    }

    // PHASE 4: DETOKENIZATION
    // ═══════════════════════
    for (size_t i = 0; i < outputIds.size(); ++i) {
        response.outputIds.push_back(outputIds[i]);
        response.outputTexts.push_back(mTokenizer->decode(outputIds[i]));
    }

    return true;
}
```

### Key Data Structures

```cpp
// File: cpp/runtime/llmRuntimeUtils.h

// REQUEST: What the user sends
struct LLMGenerationRequest {
    struct Request {
        std::vector<Message> messages;  // Chat messages
    };
    std::vector<Request> requests;      // Batch of requests

    // Sampling parameters
    float temperature{1.0f};
    float topP{1.0f};
    int64_t topK{50};
    int64_t maxGenerateLength{128};

    // Optional features
    std::string loraWeightsName{""};
    bool saveSystemPromptKVCache{false};
};

// RESPONSE: What the user gets back
struct LLMGenerationResponse {
    std::vector<std::vector<int32_t>> outputIds;   // Token IDs per request
    std::vector<std::string> outputTexts;           // Decoded text
};

// MESSAGE: A single chat turn
struct Message {
    struct MessageContent {
        std::string type;     // "text" or "image"
        std::string content;  // The actual content
    };
    std::string role;                      // "user", "assistant", "system"
    std::vector<MessageContent> contents;
};
```

---

## 2. The Decode Loop

### Prefill vs Decode Execution

```cpp
// File: cpp/runtime/llmEngineRunner.cpp

// PREFILL: Process all input tokens at once
bool LLMEngineRunner::executePrefillStep(
    rt::Tensor const& inputIds,        // [batch, seq_len] - ALL input tokens
    rt::Tensor const& contextLengths,  // [batch] - length of each sequence
    rt::Tensor& outputLogits,          // [batch, vocab_size] - output
    cudaStream_t stream)
{
    // 1. Bind input tensors to engine
    mPrefillExecutionContext->setTensorAddress(
        "input_ids",
        const_cast<void*>(inputIds.rawPointer())
    );

    // 2. Bind KV cache (where to WRITE new K, V values)
    bindKVCacheToEngine(activeBatchSize);

    // 3. Execute TensorRT engine (all transformer layers)
    mPrefillExecutionContext->enqueueV3(stream);

    // 4. Update KV cache lengths
    mKVCache.commitSequenceLength(contextLengths, stream);

    return true;
}


// DECODE: Process one new token at a time
bool LLMEngineRunner::executeVanillaDecodingStep(
    rt::Tensor const& inputIds,   // [batch, 1] - JUST the new token
    rt::Tensor& outputLogits,     // [batch, vocab_size]
    cudaStream_t stream)
{
    // 1. Update context lengths (we're adding 1 token)
    kernel::incrementLengthTensor(mSequenceContextLengths, 1, stream);

    // 2. Check if we have a captured CUDA graph
    size_t configHash = computeConfigHash(inputIds.getShape(), mActiveLoraWeightsName);

    if (mCudaGraphs.count(configHash) > 0) {
        // FAST PATH: Launch pre-captured graph
        auto& [graph, graphExec] = mCudaGraphs[configHash];
        cudaGraphLaunch(graphExec, stream);
    } else {
        // SLOW PATH: Regular execution
        mGenerationExecutionContext->setTensorAddress("input_ids", ...);
        mGenerationExecutionContext->enqueueV3(stream);
    }

    // 3. Update KV cache (we wrote 1 new token)
    mKVCache.commitSequenceLength(1, stream);

    return true;
}
```

### The Decode State Machine

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DECODE STATE MACHINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                          ┌───────────────┐                                  │
│                          │    START      │                                  │
│                          └───────┬───────┘                                  │
│                                  │                                          │
│                                  ▼                                          │
│                          ┌───────────────┐                                  │
│                          │   TOKENIZE    │                                  │
│                          │  input text   │                                  │
│                          └───────┬───────┘                                  │
│                                  │                                          │
│                                  ▼                                          │
│                          ┌───────────────┐                                  │
│                          │    PREFILL    │                                  │
│                          │ (all tokens)  │──────────────┐                   │
│                          └───────┬───────┘              │                   │
│                                  │                      │ fill              │
│                                  ▼                      ▼                   │
│                          ┌───────────────┐      ┌─────────────┐            │
│              ┌──────────►│    SAMPLE     │      │  KV CACHE   │            │
│              │           │  next token   │      └─────────────┘            │
│              │           └───────┬───────┘              ▲                   │
│              │                   │                      │                   │
│              │                   ▼                      │ append            │
│              │           ┌───────────────┐              │                   │
│              │           │    DECODE     │──────────────┘                   │
│              │           │  (1 token)    │                                  │
│              │           └───────┬───────┘                                  │
│              │                   │                                          │
│              │                   ▼                                          │
│              │           ┌───────────────┐                                  │
│              │      NO   │   EOS or      │                                  │
│              └───────────│  max length?  │                                  │
│                          └───────┬───────┘                                  │
│                                  │ YES                                      │
│                                  ▼                                          │
│                          ┌───────────────┐                                  │
│                          │  DETOKENIZE   │                                  │
│                          │   output      │                                  │
│                          └───────┬───────┘                                  │
│                                  │                                          │
│                                  ▼                                          │
│                          ┌───────────────┐                                  │
│                          │     END       │                                  │
│                          └───────────────┘                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Hands-on Exercise: Trace Decode Step

```cpp
// Exercise: Add timing to understand where time goes

// File: cpp/runtime/llmEngineRunner.cpp

bool LLMEngineRunner::executeVanillaDecodingStep(...) {
    auto start = std::chrono::high_resolution_clock::now();

    // Time 1: Context length update
    auto t1 = std::chrono::high_resolution_clock::now();
    kernel::incrementLengthTensor(mSequenceContextLengths, 1, stream);
    cudaStreamSynchronize(stream);  // Force timing
    auto t2 = std::chrono::high_resolution_clock::now();

    // Time 2: Tensor binding
    mGenerationExecutionContext->setTensorAddress(...);
    auto t3 = std::chrono::high_resolution_clock::now();

    // Time 3: Engine execution (THE BIG ONE)
    mGenerationExecutionContext->enqueueV3(stream);
    cudaStreamSynchronize(stream);
    auto t4 = std::chrono::high_resolution_clock::now();

    // Time 4: KV cache update
    mKVCache.commitSequenceLength(1, stream);
    cudaStreamSynchronize(stream);
    auto t5 = std::chrono::high_resolution_clock::now();

    // Print timings
    LOG_INFO("Context update: %d us", duration_cast<microseconds>(t2-t1).count());
    LOG_INFO("Tensor binding: %d us", duration_cast<microseconds>(t3-t2).count());
    LOG_INFO("Engine execute: %d us", duration_cast<microseconds>(t4-t3).count());
    LOG_INFO("KV cache update: %d us", duration_cast<microseconds>(t5-t4).count());

    return true;
}

// Expected output (approximate):
// Context update: 50 us
// Tensor binding: 10 us
// Engine execute: 30000 us  ◄── 99% of time is here!
// KV cache update: 100 us
```

---

## 3. Attention Kernels

### Two Types of Attention

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CONTEXT vs DECODE ATTENTION                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CONTEXT ATTENTION (Prefill)                                                │
│  ═══════════════════════════                                                │
│  File: cpp/kernels/contextAttentionKernels/                                 │
│                                                                             │
│  Input:                                                                     │
│    Q: [batch, seq_len, heads, head_dim]  ← Many queries                    │
│    K: [batch, seq_len, heads, head_dim]  ← Many keys                       │
│    V: [batch, seq_len, heads, head_dim]  ← Many values                     │
│                                                                             │
│  Characteristics:                                                           │
│    • Processes ALL input tokens at once                                     │
│    • Highly parallel (GPU-friendly)                                         │
│    • Uses Flash Attention (FMHA) for efficiency                            │
│    • Computes Q×K^T for all pairs                                          │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │   Q      K^T     =    Attention Matrix                      │           │
│  │ ┌───┐   ┌───┐       ┌─────────────────┐                    │           │
│  │ │ q0│   │k0 k1│     │ q0·k0  q0·k1 ...│                    │           │
│  │ │ q1│ × │k0 k1│  =  │ q1·k0  q1·k1 ...│                    │           │
│  │ │...│   │... .│     │  ...    ...     │                    │           │
│  │ └───┘   └───┘       └─────────────────┘                    │           │
│  │ [N,d]   [d,N]           [N,N]                              │           │
│  │                                                             │           │
│  │ All N×N pairs computed in parallel!                        │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                                                                             │
│                                                                             │
│  DECODE ATTENTION (Generation)                                              │
│  ═════════════════════════════                                              │
│  File: cpp/kernels/decodeAttentionKernels/                                  │
│                                                                             │
│  Input:                                                                     │
│    Q: [batch, 1, heads, head_dim]        ← Just ONE new query              │
│    K: [batch, seq_len, heads, head_dim]  ← All cached keys                 │
│    V: [batch, seq_len, heads, head_dim]  ← All cached values               │
│                                                                             │
│  Characteristics:                                                           │
│    • Processes ONE new token                                                │
│    • Q is single vector, K/V are from cache                                │
│    • Uses XQA (Extended Query Attention) kernel                            │
│    • Computes Q×K^T for just new token vs all cached                       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │   Q      K^T     =    Attention Vector                      │           │
│  │ ┌───┐   ┌───────┐     ┌─────────────────┐                  │           │
│  │ │ qN│ × │k0 k1..│  =  │ qN·k0 qN·k1 ... │                  │           │
│  │ └───┘   └───────┘     └─────────────────┘                  │           │
│  │ [1,d]   [d,N]             [1,N]                            │           │
│  │                                                             │           │
│  │ Only N dot products (not N²)!                              │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The XQA Kernel (Decode Attention)

```cpp
// File: cpp/kernels/decodeAttentionKernels/decoderXQARunner.h

struct XQALaunchParams {
    void* output;               // Where to write result
    void const* qInputPtr;      // Query (just 1 token)

    struct KVCache {
        void* data;             // All cached K, V
        int32_t const* lengths; // Current seq lengths
        uint32_t capacity;      // Max capacity
    } kvCache;

    int32_t numQheads;          // Query heads (e.g., 32)
    int32_t numKVheads;         // KV heads (e.g., 8 for GQA)
    int32_t headSize;           // Dimension per head (e.g., 128)
    int32_t batchSize;
};

// The kernel itself (simplified)
__global__ void decodeAttentionKernel(XQALaunchParams params) {
    // Each block handles one batch element
    int batchIdx = blockIdx.x;

    // Load query for this position
    half* q = load_query(params.qInputPtr, batchIdx);

    // Compute attention scores against ALL cached keys
    float scores[MAX_SEQ_LEN];
    for (int pos = 0; pos < params.kvCache.lengths[batchIdx]; pos++) {
        half* k = load_key_from_cache(params.kvCache.data, batchIdx, pos);
        scores[pos] = dot_product(q, k, params.headSize);
    }

    // Softmax
    softmax_inplace(scores, params.kvCache.lengths[batchIdx]);

    // Weighted sum of values
    half output[HEAD_SIZE] = {0};
    for (int pos = 0; pos < params.kvCache.lengths[batchIdx]; pos++) {
        half* v = load_value_from_cache(params.kvCache.data, batchIdx, pos);
        for (int d = 0; d < params.headSize; d++) {
            output[d] += scores[pos] * v[d];
        }
    }

    // Write output
    store_output(params.output, batchIdx, output);
}
```

### Why Attention is the Bottleneck

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ATTENTION MEMORY ACCESS PATTERN                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  For EACH decode step, we must:                                             │
│                                                                             │
│  1. Load Q (new token):        128 dims × 2 bytes = 256 bytes              │
│  2. Load ALL cached K:         seq_len × 128 × 2 = seq_len × 256 bytes     │
│  3. Load ALL cached V:         seq_len × 128 × 2 = seq_len × 256 bytes     │
│  4. Write output:              128 × 2 = 256 bytes                          │
│                                                                             │
│  EXAMPLE: seq_len = 2000, 32 layers, 8 KV heads                            │
│  ═════════════════════════════════════════════════════════════════════════ │
│                                                                             │
│  Per layer:   256 + (2000 × 256) + (2000 × 256) + 256                      │
│             = 1,024,512 bytes = ~1 MB                                       │
│                                                                             │
│  All layers:  32 × 1 MB = 32 MB                                            │
│                                                                             │
│  At 900 GB/s bandwidth:  32 MB / 900 GB/s = 35 μs                          │
│                                                                             │
│  ACTUAL TIME: ~30 ms                                                        │
│                                                                             │
│  WHERE'S THE REST? Compute overhead, synchronization, kernel launch        │
│                                                                             │
│  KEY INSIGHT: Even with perfect memory access, decode attention            │
│               is fundamentally limited by memory bandwidth                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. CUDA Graphs

### The Problem CUDA Graphs Solve

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         WITHOUT CUDA GRAPHS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CPU                          GPU                                           │
│  ───                          ───                                           │
│  │                                                                          │
│  │ Prepare kernel 1 ─────────────┐                                         │
│  │ (5 μs CPU work)               │                                         │
│  │                               ▼                                         │
│  │                            [Execute kernel 1]                            │
│  │                               │                                         │
│  │ Prepare kernel 2 ◄────────────┘                                         │
│  │ (5 μs CPU work)               │                                         │
│  │                               ▼                                         │
│  │                            [Execute kernel 2]                            │
│  │                               │                                         │
│  │ Prepare kernel 3 ◄────────────┘                                         │
│  │ ...                                                                      │
│  │                                                                          │
│  │ For 100 kernels: 100 × 5 μs = 500 μs CPU overhead                       │
│  │ On edge device with slow CPU: could be 5+ ms!                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          WITH CUDA GRAPHS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CAPTURE PHASE (once):                                                      │
│  ═════════════════════                                                      │
│  CPU records: kernel1 → kernel2 → kernel3 → ... → kernelN                  │
│  Saves as graph in GPU memory                                               │
│                                                                             │
│  REPLAY PHASE (every decode step):                                          │
│  ═════════════════════════════════                                          │
│  CPU                          GPU                                           │
│  ───                          ───                                           │
│  │                                                                          │
│  │ cudaGraphLaunch() ───────────────┐                                      │
│  │ (single call!)                   │                                      │
│  │                                  ▼                                      │
│  │                            [kernel 1]                                    │
│  │                            [kernel 2]                                    │
│  │                            [kernel 3]  All pre-recorded!                │
│  │                            ...                                           │
│  │                            [kernel N]                                    │
│  │                                  │                                      │
│  │ Done! ◄──────────────────────────┘                                      │
│  │                                                                          │
│  │ CPU overhead: ~10 μs (one call instead of 100)                          │
│  │ Speedup: 50× less CPU time                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### CUDA Graph Implementation

```cpp
// File: cpp/runtime/llmEngineRunner.cpp

bool LLMEngineRunner::captureVanillaDecodingCudaGraph(
    rt::Tensor const& inputIds,
    rt::Tensor& outputLogits,
    std::string const& loraWeightsName,
    cudaStream_t stream)
{
    // 1. Compute unique hash for this configuration
    size_t configHash = 0;
    hash_utils::hashCombine(configHash, inputIds.getShape()[0]);  // batch size
    hash_utils::hashCombine(configHash, loraWeightsName);

    // 2. Check if already captured
    if (mCudaGraphs.count(configHash) > 0) {
        return true;  // Already have this graph
    }

    // 3. Start capture
    cudaGraph_t graph;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    // 4. Execute the operations we want to capture
    mGenerationExecutionContext->setTensorAddress("input_ids", inputIds.rawPointer());
    mGenerationExecutionContext->setTensorAddress("output_logits", outputLogits.rawPointer());
    // ... bind all other tensors ...
    mGenerationExecutionContext->enqueueV3(stream);

    // 5. End capture
    cudaStreamEndCapture(stream, &graph);

    // 6. Create executable graph
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

    // 7. Store for later use
    mCudaGraphs[configHash] = {graph, graphExec};

    return true;
}
```

### When Graphs Are Used

```cpp
// In executeVanillaDecodingStep:

bool LLMEngineRunner::executeVanillaDecodingStep(...) {
    size_t configHash = computeConfigHash(...);

    auto it = mCudaGraphs.find(configHash);
    if (it != mCudaGraphs.end()) {
        // GRAPH PATH: Fast!
        cudaGraphLaunch(it->second.second, stream);
    } else {
        // REGULAR PATH: Slower, but captures graph for next time
        mGenerationExecutionContext->enqueueV3(stream);
        captureVanillaDecodingCudaGraph(...);  // Capture for future
    }
}
```

### Hands-on Exercise: Measure Graph Speedup

```cpp
// Exercise: Compare execution time with and without graphs

#include <chrono>

void measureGraphImpact() {
    // Warm up
    for (int i = 0; i < 10; i++) {
        runner.executeVanillaDecodingStep(...);
    }
    cudaStreamSynchronize(stream);

    // Measure WITHOUT graph (force regular path)
    runner.clearCudaGraphs();  // If you add this method
    auto start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        runner.executeVanillaDecodingStep(...);
    }
    cudaStreamSynchronize(stream);
    auto end1 = std::chrono::high_resolution_clock::now();

    // Measure WITH graph (second run uses captured graph)
    auto start2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        runner.executeVanillaDecodingStep(...);
    }
    cudaStreamSynchronize(stream);
    auto end2 = std::chrono::high_resolution_clock::now();

    printf("Without graph: %lld us per step\n",
           duration_cast<microseconds>(end1 - start1).count() / 100);
    printf("With graph: %lld us per step\n",
           duration_cast<microseconds>(end2 - start2).count() / 100);
}

// Expected: 5-15% improvement on desktop, 20-30% on edge (slower CPU)
```

---

## Feynman Self-Test

- [ ] **What are the main phases of handleRequest()?**
  > Tokenize → Prefill → Decode loop (sample → decode → repeat) → Detokenize

- [ ] **Why is decode attention different from context attention?**
  > Decode has Q=[1] (one new token) vs K,V=[N] (all cached). Context has Q,K,V=[N] (all tokens).

- [ ] **What problem do CUDA graphs solve?**
  > Eliminate per-kernel CPU overhead by recording and replaying a sequence of GPU operations.

- [ ] **Where does most time go in a decode step?**
  > Attention kernel (~95%), specifically loading K,V from cache for all previous tokens.

- [ ] **Why is LLM inference memory-bound during decode?**
  > Must load all cached K,V (grows linearly with sequence) but only computes O(1) new values.

## If You're Stuck

### "Can't find the kernel code"
```bash
grep -r "decodeAttention" cpp/kernels/
grep -r "__global__" cpp/kernels/  # Find all CUDA kernels
```

### "Don't understand the execution flow"
Add logging:
```cpp
LOG_INFO("Entering executePrefillStep");
// ... function body ...
LOG_INFO("Exiting executePrefillStep");
```

### "CUDA graphs don't seem faster"
- Ensure you're measuring after warm-up
- Check that graph capture succeeded
- On desktop GPUs, improvement may be small (fast CPU)

---

## What's Next?

You now understand:
- ✅ Runtime class hierarchy and responsibilities
- ✅ The decode loop state machine
- ✅ Context vs decode attention kernels
- ✅ CUDA graph optimization

**Next**: [06 Optimization Techniques](06_optimizations.md) - Deep dive into all optimization techniques used in TensorRT-Edge-LLM.

---

*← [04 Python Pipeline](04_python_pipeline.md) | [06 Optimizations →](06_optimizations.md)*
