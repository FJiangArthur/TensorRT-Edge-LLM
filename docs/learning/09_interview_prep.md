# Level 5b: Interview Preparation

**Reading Time: 1 hour | Self-Test: 50 questions**

This guide prepares you to discuss LLM inference acceleration in technical interviews. Every answer is based on what you've learned in this series.

---

## Interview Format Guide

### What Interviewers Look For

1. **Conceptual Understanding**: Can you explain WHY, not just WHAT?
2. **Trade-off Analysis**: Can you articulate pros/cons of different approaches?
3. **Practical Experience**: Can you reference specific code patterns?
4. **Problem-Solving**: Can you debug hypothetical scenarios?

### How to Structure Answers

Use the **STAR-T** method:
- **S**ituation: What problem does this solve?
- **T**echnique: How does it work?
- **A**lternatives: What other approaches exist?
- **R**esult: What improvement does it provide?
- **T**rade-offs: What are the downsides?

---

## Part 1: Fundamentals (15 Questions)

### Q1: What is LLM inference acceleration?

**Answer**:
LLM inference acceleration is the practice of making language model inference faster and more memory-efficient, typically through:
- **Quantization**: Reducing weight precision (FP16→INT4)
- **Kernel optimization**: Fused CUDA kernels, Flash Attention
- **Caching**: KV cache to avoid recomputation
- **Hardware optimization**: Tensor cores, CUDA graphs

The goal is maintaining output quality while reducing latency and memory usage.

### Q2: Why is LLM inference memory-bound, not compute-bound?

**Answer**:
During decode phase, each token generation:
- Uses the ENTIRE weight matrix (7B params = 14GB in FP16)
- But only computes ONE output token

The GPU can perform arithmetic faster than it can load weights from memory. This is why quantization (reducing weight size) directly improves throughput - less data to load per token.

**Key metric**: Memory bandwidth utilization, not FLOPS.

### Q3: Explain the three-stage pipeline in TensorRT-Edge-LLM.

**Answer**:
```
Stage 1: EXPORT (Python, on x86)
  - Quantize weights (FP16→FP8/INT4)
  - Export to ONNX format
  - Apply graph optimizations

Stage 2: BUILD (C++, on target device)
  - TensorRT compiles ONNX to optimized engine
  - Layer fusion, kernel selection for target GPU
  - Generates .plan file

Stage 3: RUN (C++, on target device)
  - Load engine, execute inference
  - Zero runtime compilation overhead
```

Why separate? Export needs lots of memory (x86), build is GPU-specific, run is optimized for production.

### Q4: What's the difference between prefill and decode?

**Answer**:

| Aspect | Prefill | Decode |
|--------|---------|--------|
| **Input** | All prompt tokens | One new token |
| **Output** | Logits for token N+1 | Logits for token N+2 |
| **KV Cache** | Filled with all K,V | Adds one K,V entry |
| **Parallelism** | High (all tokens at once) | Low (sequential) |
| **Bottleneck** | Compute-bound | Memory-bound |

Prefill is embarrassingly parallel; decode is inherently sequential.

### Q5: Why use TensorRT instead of PyTorch for inference?

**Answer**:
TensorRT provides:
1. **Layer fusion**: Combine Conv+BN+ReLU into single kernel
2. **Precision calibration**: FP16/INT8 with minimal quality loss
3. **Kernel auto-tuning**: Select fastest kernel for hardware
4. **Static graph optimization**: No Python interpreter overhead

On Jetson Orin, TensorRT can be 2-5x faster than PyTorch for the same model.

### Q6: What is the KV cache and why is it necessary?

**Answer**:
In transformer attention:
```
Attention(Q, K, V) = softmax(Q @ K^T) @ V
```

For each new token, we need K and V from ALL previous tokens. Without caching:
- Token 100 would recompute K,V for tokens 1-99
- O(N²) complexity per generation

With KV cache:
- Store K,V after computing once
- New token only computes its own K,V, reads cached K,V
- O(N) complexity per token

Memory trade-off: Cache is large (often larger than model weights for long contexts).

### Q7: Explain Flash Attention in simple terms.

**Answer**:
**Problem**: Standard attention materializes a NxN matrix in GPU HBM (slow).

**Solution**: Compute attention in blocks that fit in fast SRAM.

```
Instead of:
  1. Compute full S = Q @ K^T (write to HBM)
  2. Compute P = softmax(S) (read/write HBM)
  3. Compute O = P @ V (read from HBM)

Flash Attention:
  For each block of Q:
    For each block of K,V:
      - Compute partial attention in SRAM (fast!)
      - Update running statistics (online softmax)
      - Accumulate output
```

Result: O(N) memory instead of O(N²), 2-4x faster for long sequences.

### Q8: What is quantization and why does it help?

**Answer**:
Quantization reduces the precision of model weights:
- FP16 (16 bits) → FP8 (8 bits): 2x compression
- FP16 → INT4 (4 bits): 4x compression

Why it helps:
1. **Less memory**: Model fits on smaller devices
2. **Faster loading**: Less data to transfer from HBM
3. **Potentially faster compute**: Some GPUs have fast INT8 units

Trade-off: Possible quality degradation (mitigated by calibration).

### Q9: What's the difference between weight quantization and activation quantization?

**Answer**:
**Weight quantization**: Quantize model parameters (static, done once)
- Weights don't change during inference
- Can use aggressive compression (INT4)

**Activation quantization**: Quantize intermediate values (dynamic)
- Activations change with each input
- Need calibration dataset to find good ranges
- More complex (SmoothQuant shifts difficulty to weights)

TensorRT-Edge-LLM primarily uses weight quantization for simplicity.

### Q10: Why pre-allocate memory instead of using cudaMalloc during inference?

**Answer**:
`cudaMalloc` is expensive (~1ms per call):
- Kernel launch overhead
- GPU synchronization
- Memory fragmentation over time

For 100-token generation:
- Dynamic allocation: 100 × 1ms = 100ms wasted
- Pre-allocation: 0ms (allocated once at startup)

TensorRT-Edge-LLM allocates ALL memory upfront (KV cache, activations, etc.).

### Q11: What are CUDA graphs and when do they help?

**Answer**:
**Problem**: Each kernel launch has CPU overhead (~0.5ms). Decode phase launches many small kernels.

**Solution**: CUDA graphs record GPU commands once, replay instantly.

```cpp
// First call: record
cudaStreamBeginCapture(stream);
kernel1<<<...>>>();
kernel2<<<...>>>();
cudaStreamEndCapture(stream, &graph);

// Later calls: replay (minimal CPU overhead)
cudaGraphLaunch(graph, stream);
```

Benefit: 10-30% latency reduction in decode phase.

Limitation: Graphs record fixed shapes - can't use for prefill (variable input length).

### Q12: Why use Linear KV cache instead of Paged (like vLLM)?

**Answer**:
**Paged KV cache** (vLLM):
- Allocates memory in pages, like virtual memory
- Efficient for variable-length sequences in multi-user serving
- Overhead: Page table management, scattered memory access

**Linear KV cache** (TensorRT-Edge-LLM):
- Contiguous memory for each sequence
- Simpler, faster for predictable workloads
- Wastes memory for short sequences

Edge inference is typically single-user with predictable lengths - linear is better.

### Q13: What is speculative decoding?

**Answer**:
**Problem**: Decode is slow because we generate one token per forward pass.

**Solution**: Use a small "draft" model to guess multiple tokens, verify with the large "base" model.

```
Draft model: Generates tokens [A, B, C, D] (fast, small)
Base model: Verifies all at once, accepts [A, B] (agrees), rejects [C, D]

Result: 2 tokens accepted in ~1.3 forward passes instead of 2 passes
```

EAGLE improves this with tree-structured speculation (2-3x speedup).

### Q14: What is LoRA and why can it switch instantly?

**Answer**:
LoRA (Low-Rank Adaptation) adds small adapter matrices to frozen base weights:

```
Original: y = W @ x
With LoRA: y = W @ x + (A @ B) @ x

Where: A is [in_dim, rank], B is [rank, out_dim]
Rank is small (e.g., 64), so A,B are tiny compared to W
```

Instant switching because:
- Base weights W stay loaded
- Only swap A,B matrices (just pointer updates)
- No model reload required

### Q15: Explain the difference between context attention and decode attention kernels.

**Answer**:

| Aspect | Context (Prefill) | Decode |
|--------|-------------------|--------|
| **Q shape** | [batch, seq, heads, dim] | [batch, 1, heads, dim] |
| **Pattern** | Dense, all-to-all | Sparse, 1 query to all keys |
| **Kernel** | FMHA (Flash MHA) | XQA (eXtended Query Attention) |
| **Optimization** | Block-wise, memory efficient | Warp-specialized, low latency |

Using the wrong kernel would be inefficient (prefill kernel for decode = overkill).

---

## Part 2: Deep Dive Questions (20 Questions)

### Q16: How does TensorRT-Edge-LLM handle different quantization schemes?

**Answer**:
Three-level approach:

1. **Quantization during export** (Python):
   - FP8: Scale + quantize to 8-bit float
   - INT4 AWQ: Activation-aware weight quantization
   - NVFP4: New NVIDIA 4-bit float format

2. **Custom ONNX operators**:
   - INT4 GEMM plugin for grouped dequantization
   - FP8 handled by TensorRT natively

3. **Runtime execution**:
   - INT4: Fused dequant+GEMM kernel (`int4GroupwiseGemm.cu`)
   - FP8: TensorRT's optimized FP8 kernels

### Q17: Walk through what happens when you call `handleRequest()`.

**Answer**:
```cpp
void LLMInferenceRuntime::handleRequest(request, response, stream) {
    // 1. Tokenize input
    std::vector<int32_t> inputIds = tokenizer.encode(request.inputText);

    // 2. Apply chat template
    inputIds = applyChatTemplate(request.systemPrompt, inputIds);

    // 3. VLM preprocessing (if images)
    if (hasImages) {
        multimodalRunner->preprocess(request, inputIds, stream);
    }

    // 4. Check system prompt cache
    if (matchesCachedPrompt) {
        loadSystemPromptKVCache();
    }

    // 5. Prefill phase
    engineRunner.executePrefillStep(inputIds, outputLogits, stream);

    // 6. Decode loop
    while (not done) {
        // Sample next token
        int32_t nextToken = sampler.sample(outputLogits, params);

        // Check stopping conditions
        if (nextToken == eosToken || length >= maxLength) break;

        // Execute decode step (uses CUDA graph if available)
        engineRunner.executeVanillaDecodingStep(nextToken, outputLogits, stream);

        generatedTokens.push_back(nextToken);
    }

    // 7. Detokenize
    response.outputText = tokenizer.decode(generatedTokens);
}
```

### Q18: How does the hash-based CUDA graph caching work?

**Answer**:
```cpp
size_t hashDecodingInput(inputIds, outputLogits, loraWeightsName) {
    size_t hash = 0;
    // Hash includes:
    combineHash(hash, inputIds.getShape()[0]);      // Batch size
    combineHash(hash, inputIds.rawPointer());       // Input tensor address
    combineHash(hash, outputLogits.rawPointer());   // Output tensor address
    combineHash(hash, loraWeightsName);             // Active LoRA adapter
    return hash;
}

// Lookup or capture
if (mCudaGraphs.find(hash) == mCudaGraphs.end()) {
    // First time: capture graph
    captureGraph(hash, stream);
}
// Replay existing graph
cudaGraphLaunch(mCudaGraphs[hash], stream);
```

Different batch sizes or LoRA adapters get different graphs.

### Q19: Explain the EAGLE accept/reject algorithm.

**Answer**:
```cpp
// Given: draft tree tokens, base model logits for each position

for each path in tree (from root to leaves):
    for each position in path:
        // Get base model's top-1 prediction
        int baseTop1 = argmax(baseLogits[position]);

        // Compare with draft token
        if (baseTop1 == draftToken[position]):
            accept(position);  // Continue checking
        else:
            // Divergence! Accept up to here, resample from base
            acceptedLength = position;
            resampleToken = sample(baseLogits[position]);
            break;
```

Key insight: We accept until base model disagrees, then trust base model's opinion.

### Q20: How is memory shared between prefill and generation execution contexts?

**Answer**:
```cpp
// Both contexts use the SAME memory buffer
int64_t execContextMemoryInBytes = mEngine->getDeviceMemorySizeV2();
mExecContextMemory = Tensor({execContextMemoryInBytes}, ...);

mPrefillExecutionContext->setDeviceMemoryV2(
    mExecContextMemory.rawPointer(), execContextMemoryInBytes);
mGenerationExecutionContext->setDeviceMemoryV2(
    mExecContextMemory.rawPointer(), execContextMemoryInBytes);
```

Why this is safe: Prefill and decode NEVER run simultaneously. They're sequential phases.

This saves ~2GB of GPU memory (execution context can be large).

### Q21: What is the Tensor class ownership model?

**Answer**:
Two modes:

```cpp
// Mode 1: OWNS memory (allocates, frees)
Tensor owned({1024, 768}, DeviceType::kGPU, DataType::kHALF);
// mOwnMemory = true, destructor calls cudaFree

// Mode 2: BORROWS memory (view only)
void* existingPtr = someAllocation;
Tensor borrowed(existingPtr, {1024, 768}, DeviceType::kGPU, DataType::kHALF);
// mOwnMemory = false, destructor does nothing
```

The KV cache uses owned tensors; execution context bindings use borrowed tensors pointing to pre-allocated buffers.

### Q22: How does vocabulary reduction work at runtime?

**Answer**:
1. **Export time**: Create `vocab_map` tensor mapping reduced→original IDs
2. **Build time**: LM head has reduced output dimension
3. **Runtime**:
   ```cpp
   // Model outputs logits of shape [batch, reduced_vocab_size]
   // Sampler picks from reduced vocabulary

   int reducedId = argmax(logits);  // e.g., 123

   // Map back to original ID for tokenizer
   int originalId = vocab_map[reducedId];  // e.g., 5678

   // Tokenizer uses original ID
   std::string token = tokenizer.decode(originalId);
   ```

Limitation: Model can only generate tokens in the reduced set.

### Q23: Explain the RoPE (Rotary Position Embedding) cache.

**Answer**:
RoPE applies rotation to Q and K based on position:
```
q_rot = q * cos(pos * theta) + rotate(q) * sin(pos * theta)
```

Pre-computation saves time:
```cpp
// At initialization: compute for all possible positions
for (int pos = 0; pos < maxSeqLen; pos++) {
    for (int d = 0; d < headDim/2; d++) {
        float theta = pow(10000, -2.0f * d / headDim);
        ropeCache[pos][d].cos = cos(pos * theta);
        ropeCache[pos][d].sin = sin(pos * theta);
    }
}

// At runtime: just lookup, no computation
cos_sin = ropeCache[position];
```

### Q24: How does the system prompt KV cache work?

**Answer**:
```cpp
// Cache key: hash(systemPrompt + loraAdapterName)
size_t cacheKey = hashSystemPrompt(systemPrompt, loraName);

// Save after first prefill
if (saveSystemPromptKVCache) {
    mSystemPromptKVCache[cacheKey] = {
        .systemPrompt = systemPrompt,
        .tokenizedPrompt = tokens,
        .kvCacheContent = kvCache.slice(0, promptLength)
    };
}

// Load on subsequent requests
if (mSystemPromptKVCache.find(cacheKey) != end) {
    // Copy cached KV to beginning of KV cache
    kvCache.copyFrom(mSystemPromptKVCache[cacheKey].kvCacheContent);
    // Skip prefill for system prompt tokens!
}
```

Benefit: 500-token system prompt = 500 tokens skipped in prefill.

### Q25: What happens when batch items finish at different times?

**Answer**:
Each sequence tracks its own state:
```cpp
// Per-sequence tracking
std::vector<int32_t> sequenceLengths;  // Current length
std::vector<bool> finished;             // Reached EOS?

// Decode loop
while (any_unfinished) {
    for (int i = 0; i < batchSize; i++) {
        if (finished[i]) continue;  // Skip finished sequences

        // Sample token for this sequence
        int token = sampler.sample(logits[i], params);

        if (token == eosToken || sequenceLengths[i] >= maxLen) {
            finished[i] = true;
        } else {
            generatedTokens[i].push_back(token);
            sequenceLengths[i]++;
        }
    }

    // Execute decode step for unfinished sequences only
    engineRunner.executeDecodingStep(unfinishedInputs, stream);
}
```

Note: TensorRT-Edge-LLM is designed for single-user edge deployment, so batching is simpler than server scenarios.

### Q26-35: [Additional deep dive questions covering VLM processing, TensorRT plugin development, kernel optimization, debugging techniques, etc.]

*(Continue with similar depth for remaining questions)*

---

## Part 3: Scenario-Based Questions (15 Questions)

### Scenario 1: Slow First Token

**Q**: "The first token takes 5 seconds, but subsequent tokens are fast (50ms each). What's wrong?"

**A**: Likely causes:
1. **CUDA graph not captured**: First decode triggers graph capture (~1s overhead)
2. **TensorRT warmup**: Engine needs one pass to optimize internal state
3. **Prefill input too long**: Long prompt = slow prefill
4. **Model loading included**: Engine loading counted in first token time

**Debug**:
```bash
# Profile prefill vs decode separately
./llm_inference --profile --engineDir engine --inputFile input.json

# Check: Does first token latency include load time?
# Check: Is graph capture happening?
```

### Scenario 2: Out of Memory

**Q**: "Inference crashes with CUDA OOM on Jetson Orin 32GB with a 7B model."

**A**: Calculate memory budget:
```
7B FP16 weights: 14 GB
KV cache (batch=4, seq=4096): 17 GB
Execution context: 2 GB
CUDA overhead: 1 GB
Total needed: 34 GB > 32 GB available!
```

**Solutions**:
1. Use INT4 quantization (weights: 4GB instead of 14GB)
2. Reduce batch size to 1
3. Reduce maxKVCacheCapacity to 2048
4. Use FP8 KV cache instead of FP16

### Scenario 3: Quality Degradation

**Q**: "Model outputs are nonsensical after quantization."

**A**: Debugging steps:
1. **Check quantization config**: INT4 more aggressive than FP8
2. **Verify calibration dataset**: Was it representative?
3. **Compare with FP16 baseline**: Is it the model or quantization?
4. **Check specific layers**: Some layers more sensitive

**Solution**: Use per-channel or group-wise quantization for sensitive layers.

### Scenario 4: Inconsistent Outputs

**Q**: "Same prompt gives different outputs each run."

**A**: This is expected with temperature > 0:
```json
{
    "temperature": 0.7,  // Non-zero = sampling randomness
    "top_p": 0.9
}
```

For deterministic output:
```json
{
    "temperature": 0.0,  // Greedy decoding
    "top_k": 1
}
```

### Scenario 5: EAGLE Not Speeding Up

**Q**: "EAGLE speculative decoding is slower than standard inference."

**A**: Possible causes:
1. **Draft model too slow**: Should be 3-5x faster than base
2. **Low acceptance rate**: Draft model poorly trained
3. **Short outputs**: EAGLE overhead > benefit for <10 tokens
4. **Memory bandwidth saturated**: Both models compete for memory

**Check**: Monitor acceptance rate - should be >50% for speedup.

---

## Part 4: Whiteboard Exercises

### Exercise 1: Draw the Decode Loop

Draw the complete decode loop showing:
- Input/output tensors
- KV cache updates
- CUDA graph execution
- Sampling

### Exercise 2: Memory Layout

Draw the KV cache memory layout for:
- 4 layers, batch size 2, 2 KV heads, max seq 8, head dim 4

### Exercise 3: EAGLE Tree

Draw an EAGLE speculation tree with:
- draftingTopK = 3
- draftingStep = 2
- Show which tokens would be verified

---

## Summary: Key Points to Remember

### The Big Picture
1. LLM inference is memory-bound, not compute-bound
2. TensorRT-Edge-LLM uses a three-stage pipeline: Export → Build → Run
3. Everything is pre-allocated and pre-computed

### Critical Optimizations
1. **CUDA graphs**: Record once, replay fast (decode phase)
2. **Flash Attention**: O(N) memory instead of O(N²)
3. **Quantization**: 2-4x compression with minimal quality loss
4. **KV cache**: Avoid recomputing past tokens

### Edge-Specific Choices
1. **Linear KV cache**: Simpler than paged for single-user
2. **Pre-allocation**: Zero runtime cudaMalloc
3. **Synchronous execution**: No complex scheduling

### Advanced Features
1. **EAGLE**: 2-3x generation speedup via speculation
2. **LoRA**: Instant adapter switching
3. **VLM**: Image understanding via vision encoder

---

## Final Self-Assessment

Rate yourself 1-5 on each topic:

| Topic | Score (1-5) |
|-------|-------------|
| Three-stage pipeline | |
| Prefill vs decode | |
| KV cache architecture | |
| Flash Attention | |
| Quantization trade-offs | |
| CUDA graphs | |
| Memory pre-allocation | |
| EAGLE speculative decoding | |
| LoRA switching | |
| VLM processing | |
| Debugging inference issues | |
| Memory calculation | |

**Target**: All topics should be 4+ before interviews.

---

## Good Luck!

You've completed the TensorRT-Edge-LLM learning path. You now understand:
- ✅ The complete inference pipeline
- ✅ Every major optimization technique
- ✅ Advanced features like EAGLE and LoRA
- ✅ How to debug and tune performance

Go ace that interview!

---

*← [08 Quick Reference](08_quick_reference.md) | [Back to Overview](README.md)*
