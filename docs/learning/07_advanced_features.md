# Level 4b: Advanced Features

**Reading Time: 2 hours | Hands-on Exercises: 5**

You've mastered the optimization techniques. Now let's explore the cutting-edge features that set TensorRT-Edge-LLM apart: speculative decoding, LoRA switching, and vision-language models.

## The Core Insight First

1. **EAGLE speculative decoding generates multiple tokens per forward pass** - draft fast, verify once
2. **LoRA adapters are hot-swappable at runtime** - one base model, many personalities
3. **VLM support processes images through separate vision encoders** - unified multimodal pipeline

## Key Numbers to Memorize

| Feature | Typical Speedup | Memory Overhead |
|---------|-----------------|-----------------|
| **EAGLE drafting** | 2-3x generation throughput | +draft model (~1GB) |
| **LoRA switching** | <1ms switch time | +rank×layers×2 weights |
| **VLM processing** | N/A (adds capability) | +vision embeddings |
| **System prompt cache** | 1.5-2x prefill speed | +cached KV for prompt |

## Table of Contents

1. [EAGLE Speculative Decoding](#1-eagle-speculative-decoding)
2. [LoRA Runtime Switching](#2-lora-runtime-switching)
3. [Vision-Language Models (VLM)](#3-vision-language-models-vlm)
4. [Combining Features](#4-combining-features)

---

## 1. EAGLE Speculative Decoding

### The Problem with Normal Decoding

Standard autoregressive generation is slow:
```
Token 1: [Forward Pass] → sample → Token 1
Token 2: [Forward Pass] → sample → Token 2
Token 3: [Forward Pass] → sample → Token 3
...
```

Each token requires a full forward pass through the model. The GPU sits idle between passes while CPU orchestrates.

### The EAGLE Solution: Draft and Verify

**Mental Model**: Think of EAGLE like having an intern (draft model) suggest multiple words, and then the expert (base model) reviews all suggestions at once.

```
┌─────────────────────────────────────────────────────────────────┐
│                     EAGLE SPECULATIVE DECODING                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PHASE 1: DRAFT (Fast, small model)                            │
│  ══════════════════════════════════                            │
│                                                                 │
│  Draft model generates a TREE of candidate tokens:             │
│                                                                 │
│                    [current]                                    │
│                        │                                        │
│           ┌───────────┼───────────┐                            │
│           │           │           │                             │
│        [the]       [and]       [but]      ← Top-K at level 1   │
│           │           │           │                             │
│     ┌─────┼─────┐    ...        ...       ← Tree expands       │
│     │     │     │                                               │
│   [cat] [dog] [big]                        ← Level 2 options   │
│                                                                 │
│  PHASE 2: VERIFY (One pass, base model)                        │
│  ══════════════════════════════════════                        │
│                                                                 │
│  Base model processes ENTIRE tree at once:                     │
│  ┌────────────────────────────────────────────────────┐        │
│  │ Verify: [current][the][cat] [current][the][dog] ...│        │
│  └────────────────────────────────────────────────────┘        │
│                                                                 │
│  PHASE 3: ACCEPT (Keep matching tokens)                        │
│  ══════════════════════════════════════                        │
│                                                                 │
│  If base model agrees: "the" ✓ "cat" ✓ → Accept 2 tokens!     │
│  If base disagrees: "the" ✓ "cat" ✗ → Accept 1, resample      │
│                                                                 │
│  Result: 2-5 tokens per "forward pass" instead of 1!           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### How EAGLE Works Step by Step

**Step 1: Prefill Both Models**
```cpp
// Base model prefill (same as normal)
baseRunner.executePrefillStep(inputIds, contextLengths, outputLogits, stream);

// Draft model prefill (uses base model hidden states!)
draftRunner.executeEaglePrefillStep(
    inputIds,
    baseHiddenStates,  // From base model - this is key!
    draftOutputLogits,
    stream
);
```

The draft model takes base model hidden states as input - it learns to predict what the base model would say!

**Step 2: Draft Tree Construction**
```cpp
for (int level = 0; level < draftingStep; level++) {
    // Draft model proposes top-K tokens at each position
    draftRunner.executeEagleDraftProposalStep(
        draftKVCache,
        treeMask,           // Which tree positions to consider
        draftingTopK,       // How many candidates per position
        draftOutputLogits,
        stream
    );

    // Select top-K tokens, expand tree
    selectTopKAndExpandTree(draftOutputLogits, tree);
}
```

**Step 3: Base Model Verification**
```cpp
// Process entire tree in ONE forward pass
baseRunner.executeEagleBaseTreeDecodingStep(
    treeTokenIds,       // All candidates packed together
    treeAttentionMask,  // Custom mask for tree structure
    baseOutputLogits,   // Logits for every tree position
    stream
);

// Run acceptance algorithm
eagleAccept(
    baseOutputLogits,    // What base model predicted
    treeTokenIds,        // What draft model proposed
    acceptedTokens,      // Output: which tokens to keep
    acceptedLength,      // Output: how many accepted
    stream
);
```

**Step 4: Accept and Continue**
```cpp
// Update both KV caches with accepted tokens
baseRunner.commitAcceptedTokens(acceptedTokens, acceptedLength, stream);
draftRunner.executeEagleAcceptDecodeTokenStep(acceptedTokens, stream);

// If 3 tokens accepted: we saved 2 forward passes!
```

### Key Files

| File | Purpose |
|------|---------|
| `cpp/runtime/llmInferenceSpecDecodeRuntime.h` | Main orchestration |
| `cpp/runtime/eagleDraftEngineRunner.h` | Draft model execution |
| `cpp/runtime/llmEngineRunner.h` | Base model tree decoding |
| `cpp/kernels/speculative/eagleAcceptKernels.h` | Accept/reject logic |

### Configuration Parameters

```json
{
  "eagle": {
    "draftingTopK": 5,         // Candidates per tree level
    "draftingStep": 3,         // Tree depth (levels)
    "verifyTreeSize": 15       // Max tokens in verification batch
  }
}
```

### Exercise 7.1: Understanding EAGLE Speedup

**Task**: Calculate theoretical EAGLE speedup.

Given:
- Draft model: 3x faster than base model
- `draftingTopK`: 5 (5 candidates per position)
- `draftingStep`: 3 (3 levels deep)
- Average acceptance rate: 60%

**Tree size**:
```
Level 0: 1 token (root)
Level 1: 5 tokens (top-5)
Level 2: 5 × 5 = 25 tokens
Level 3: Not generated (verify at level 2)

Total tree: 1 + 5 + 25 = 31 candidates
But verifyTreeSize=15, so we verify ~15 candidates
```

**Cost comparison**:

*Without EAGLE (generate 3 tokens)*:
```
Base forward passes: 3
Draft forward passes: 0
Total cost: 3 base passes
```

*With EAGLE (verify tree, accept ~3 tokens)*:
```
Draft passes: 3 (build tree) = 1 base-equivalent (3x faster)
Base passes: 1 (verify tree)
Total cost: ~2 base passes
```

**Speedup**: 3 / 2 = 1.5x for 3 tokens

**Questions**:
1. What happens if acceptance rate drops to 30%?
2. Why is batch size limited to 1 for EAGLE?
3. What if the draft model is too slow?

### Exercise 7.2: Trace EAGLE Code Flow

**Task**: Find these functions in the codebase and understand their role.

```bash
# Search for EAGLE functions
grep -rn "executeEagle" cpp/runtime/ --include="*.cpp"
grep -rn "eagleAccept" cpp/kernels/ --include="*.cu"
```

**Fill in what each function does**:

| Function | What it does |
|----------|-------------|
| `executeEaglePrefillStep()` | ____________ |
| `executeEagleDraftProposalStep()` | ____________ |
| `executeEagleBaseTreeDecodingStep()` | ____________ |
| `eagleAccept()` | ____________ |

---

## 2. LoRA Runtime Switching

### The Problem

Fine-tuning entire models is expensive:
- 7B model = 14GB weights to store per fine-tune
- 10 specializations = 140GB of storage
- Switching requires model reload (~30 seconds)

### The LoRA Solution

LoRA (Low-Rank Adaptation) adds small adapter matrices to specific layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                     LoRA ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Original linear layer:   y = W × x                            │
│                           (W is frozen, never changes)          │
│                                                                 │
│  With LoRA:               y = W × x + (A × B) × x              │
│                                        ↑                        │
│                               Small matrices!                   │
│                                                                 │
│  ┌────────────────────────────────────────────────────┐        │
│  │                                                    │        │
│  │   Input (x)                                       │        │
│  │      │                                            │        │
│  │      ├────────┬────────────┐                     │        │
│  │      │        │            │                      │        │
│  │      ▼        ▼            ▼                      │        │
│  │   ┌─────┐  ┌─────┐     ┌─────┐                   │        │
│  │   │  W  │  │ A   │     │ B   │                   │        │
│  │   │     │  │rank │     │rank │                   │        │
│  │   │4096 │  │ 64  │     │ 64  │                   │        │
│  │   │  ×  │  │  ×  │  ×  │  ×  │                   │        │
│  │   │4096 │  │4096 │     │4096 │                   │        │
│  │   └──┬──┘  └──┬──┘     └──┬──┘                   │        │
│  │      │        │            │                      │        │
│  │      │        └─────┬──────┘                      │        │
│  │      │              │                             │        │
│  │      │          ┌───▼───┐                         │        │
│  │      │          │ A × B │   ← Rank-64 adapter    │        │
│  │      │          └───┬───┘                         │        │
│  │      │              │                             │        │
│  │      └──────┬───────┘                             │        │
│  │             │                                     │        │
│  │         ┌───▼───┐                                 │        │
│  │         │  ADD  │                                 │        │
│  │         └───┬───┘                                 │        │
│  │             │                                     │        │
│  │          Output (y)                               │        │
│  │                                                   │        │
│  └────────────────────────────────────────────────────┘        │
│                                                                 │
│  Memory savings:                                               │
│  Original: 4096 × 4096 = 16M params per layer                 │
│  LoRA:     64 × 4096 × 2 = 512K params per layer (32x less!)  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Runtime Switching Implementation

**File**: `cpp/runtime/llmEngineRunner.h`

```cpp
class LLMEngineRunner {
    // Storage for multiple adapters
    std::unordered_map<std::string, std::vector<rt::Tensor>> mLoraWeights;

    // Currently active adapter
    std::string mActiveLoraWeightsName;

public:
    // Add a new adapter from file
    bool addLoraWeights(
        const std::string& name,     // "medical", "legal", etc.
        const std::string& path,     // SafeTensors file
        cudaStream_t stream
    );

    // Switch active adapter (fast!)
    bool switchLoraWeights(
        const std::string& name,     // Empty string = disable LoRA
        cudaStream_t stream
    );

    // Query available adapters
    std::vector<std::string> getAvailableLoraWeights();
    std::string getActiveLoraWeightsName();
};
```

### How Switching Works

```cpp
bool LLMEngineRunner::switchLoraWeights(
    const std::string& loraWeightsName,
    cudaStream_t stream)
{
    // Case 1: Disable LoRA (use base model)
    if (loraWeightsName.empty()) {
        resetLoraWeights(stream);  // Set dummy tensors
        mActiveLoraWeightsName = "";
        return true;
    }

    // Case 2: Find requested adapter
    auto it = mLoraWeights.find(loraWeightsName);
    if (it == mLoraWeights.end()) {
        return false;  // Adapter not loaded
    }

    // Case 3: Bind adapter tensors to engine
    auto& loraTensors = it->second;
    for (const auto& tensor : loraTensors) {
        // Update TensorRT binding to point to adapter weights
        setTensorAddress(tensor.getName(), tensor.rawPointer());
        setInputShape(tensor.getName(), tensor.getShape());
    }

    mActiveLoraWeightsName = loraWeightsName;
    return true;
}
```

### Zero-Rank Trick for Missing Layers

What if an adapter only fine-tunes some layers?

```cpp
// For layers WITHOUT adapter weights:
// Set shape to rank=1 and bind dummy tensor (zeros)

// Result: A × B = 0 (no effect on that layer)
// The layer uses base weights only
```

### CUDA Graphs and LoRA

Different adapters = different CUDA graphs!

```cpp
// Graph hash includes LoRA adapter name
size_t hashDecodingInput(..., const std::string& loraWeightsName) {
    // ... other hash components ...
    hash_combine(result, std::hash<std::string>{}(loraWeightsName));
    return result;
}

// So switching adapters triggers new graph capture
// But same adapter reuses its captured graph
```

### Exercise 7.3: LoRA Memory Calculation

**Task**: Calculate LoRA memory overhead.

Given:
- Model: 32 layers, hidden_size=4096
- LoRA applied to: q_proj, k_proj, v_proj, o_proj (4 per layer)
- LoRA rank: 64
- Precision: FP16 (2 bytes)

**Per-layer LoRA memory**:
```
Each projection: 2 matrices (A and B)
  A: [4096, 64] = 262K params
  B: [64, 4096] = 262K params

Per projection: 524K params
Per layer (4 projections): 2.1M params
```

**Total adapter size**:
```
32 layers × 2.1M params × 2 bytes = ______ MB
```

**Compare to full model**:
```
7B model in FP16 = 14GB
LoRA adapter = ______ MB
Ratio: ______
```

### Exercise 7.4: LoRA Switching Code

**Task**: Write pseudocode for a multi-adapter application.

```cpp
// Scenario: Customer service bot with domain adapters

void handleRequest(Request& req, Response& resp) {
    // Step 1: Detect domain from request
    std::string domain = detectDomain(req.text);

    // Step 2: Switch to appropriate adapter
    if (domain == "billing") {
        runner.switchLoraWeights("billing_adapter", stream);
    } else if (domain == "technical") {
        runner.switchLoraWeights("technical_adapter", stream);
    } else {
        runner.switchLoraWeights("", stream);  // Base model
    }

    // Step 3: Generate response
    runner.generate(req, resp, stream);
}
```

**Questions**:
1. What's the switching latency between adapters?
2. Does switching invalidate the KV cache?
3. Can you use different adapters for different batch items?

---

## 3. Vision-Language Models (VLM)

### The Architecture

VLMs process images and text together:

```
┌─────────────────────────────────────────────────────────────────┐
│                     VLM ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT                                                          │
│  ═════                                                          │
│                                                                 │
│  Image: [photo.jpg]    Text: "What is in this image?"          │
│            │                       │                            │
│            ▼                       ▼                            │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │ Vision Encoder  │    │   Tokenizer     │                    │
│  │ (ViT)           │    │                 │                    │
│  └────────┬────────┘    └────────┬────────┘                    │
│           │                      │                              │
│           ▼                      ▼                              │
│  Vision Embeddings        Token Embeddings                      │
│  [256 × 4096]             [20 × 4096]                          │
│                                                                 │
│  MERGE                                                          │
│  ═════                                                          │
│                                                                 │
│  Input text: "What is in <image> ?"                            │
│                           ↑                                     │
│              Replace with vision embeddings!                    │
│                                                                 │
│  Merged: [What][is][in][v1][v2]...[v256][?]                    │
│                         └───────────────┘                       │
│                       256 vision tokens                         │
│                                                                 │
│  LLM PROCESSING                                                 │
│  ══════════════                                                 │
│                                                                 │
│  ┌─────────────────────────────────────────────────────┐       │
│  │ Transformer processes all tokens (text + vision)    │       │
│  │ Attention: text attends to vision, vision to text   │       │
│  └─────────────────────────────────────────────────────┘       │
│                           │                                     │
│                           ▼                                     │
│  OUTPUT: "This image shows a cat sitting on a couch."          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Supported Models

| Model | Vision Encoder | Tokens per Image | Special Features |
|-------|---------------|------------------|------------------|
| **Qwen2-VL** | Dynamic patches | 256-576 (varies) | Adaptive resolution |
| **Qwen2.5-VL** | Enhanced ViT | 256-576 | Better spatial awareness |
| **Qwen3-VL** | Deepstack | Variable | Multi-scale features |
| **InternVL3** | Fixed 448×448 | 256 | Consistent processing |
| **Phi-4-Multimodal** | LoRA-adapted | Variable | Efficient fine-tuning |

### Implementation Flow

**File**: `cpp/multimodal/multimodalRunner.h`

**Step 1: Create Vision Runner**
```cpp
// Factory creates appropriate runner for your model
std::unique_ptr<MultimodalRunner> visionRunner =
    MultimodalRunner::create(
        visionEngineDir,     // TensorRT engine for vision model
        maxBatchSize,
        maxSequenceLength,
        stream
    );
```

**Step 2: Preprocess Images**
```cpp
bool MultimodalRunner::preprocess(
    const LLMGenerationRequest& request,
    std::vector<std::vector<int32_t>>& batchedInputIds,
    Tokenizer* tokenizer,
    Tensor& ropeCosSin,
    cudaStream_t stream)
{
    // 1. Load images from paths
    std::vector<cv::Mat> images;
    for (const auto& path : request.imagePaths) {
        images.push_back(cv::imread(path));
    }

    // 2. Resize to model's expected size
    // Qwen: Dynamic based on aspect ratio
    // InternVL: Fixed 448×448

    // 3. Normalize pixel values
    // ImageNet mean/std typically

    // 4. Run vision encoder
    infer(stream);  // TensorRT execution

    // 5. Get vision embeddings
    Tensor& visionEmbeds = getOutputEmbedding();
    // Shape: [num_images × tokens_per_image, hidden_size]

    // 6. Merge with text tokens
    // Replace <image> placeholder with vision embeddings

    return true;
}
```

**Step 3: Generate with Multimodal Input**
```cpp
// The LLM sees vision embeddings as special tokens
// KV cache stores attention for both text and vision
// Generation proceeds normally after prefill
```

### Dynamic Image Patching (Qwen)

Qwen models use dynamic patches based on image size:

```
┌─────────────────────────────────────────────────────────────────┐
│                   DYNAMIC PATCHING                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Small image (256×256):                                        │
│  ┌────────────┐                                                │
│  │ 4×4 patches│ = 16 patches = 16 × 16 = 256 tokens           │
│  └────────────┘                                                │
│                                                                 │
│  Large image (1024×512):                                       │
│  ┌────────────────────────────┐                                │
│  │                            │                                │
│  │   16×8 patches             │ = 128 patches = ~512 tokens   │
│  │                            │                                │
│  └────────────────────────────┘                                │
│                                                                 │
│  Benefits:                                                     │
│  - More detail for larger images                               │
│  - Fewer tokens for smaller images (faster)                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Exercise 7.5: VLM Memory Impact

**Task**: Calculate memory impact of adding images.

Given:
- Model: 7B with 4096 hidden size
- Image: 512×512 → 256 vision tokens
- Text prompt: 100 tokens
- KV cache: FP16

**Without image**:
```
Context length: 100 tokens
KV cache per layer: 2 × numKVHeads × seqLen × headDim × 2 bytes
```

**With image**:
```
Context length: 100 + 256 = 356 tokens
KV cache increase: (256/100) = 2.56x for prefill
```

**Questions**:
1. How does image resolution affect prefill latency?
2. Can you process multiple images in one request?
3. What happens if you exceed maxSequenceLength with images?

---

## 4. Combining Features

### Feature Compatibility Matrix

| Feature A | Feature B | Compatible? | Notes |
|-----------|-----------|-------------|-------|
| EAGLE | LoRA | Yes | Separate graphs per LoRA |
| EAGLE | VLM | Yes | Vision prefill then EAGLE decode |
| LoRA | VLM | Yes | Vision encoder unaffected |
| CUDA Graphs | EAGLE | No | Tree shapes are dynamic |
| CUDA Graphs | LoRA | Yes | Different graphs per adapter |
| System Prompt Cache | LoRA | Partial | Cache key includes LoRA name |

### Example: Medical VLM with LoRA

```cpp
// Scenario: Medical image analysis with specialized adapter

void analyzeMedicalImage(
    const std::string& imagePath,
    const std::string& query,
    std::string& analysis)
{
    // 1. Switch to medical adapter
    runtime.switchLoraWeights("radiology_lora", stream);

    // 2. Create multimodal request
    LLMGenerationRequest request;
    request.imagePaths = {imagePath};
    request.inputText = "You are a radiologist. " + query;
    request.maxLength = 500;

    // 3. Generate analysis
    LLMGenerationResponse response;
    runtime.handleRequest(request, response, stream);

    analysis = response.outputText;
}

// Usage
analyzeMedicalImage("chest_xray.jpg",
                   "Describe any abnormalities visible in this chest X-ray.",
                   result);
```

### Example: Multi-Domain Chatbot with EAGLE

```cpp
// High-throughput chatbot with speculative decoding

void handleChatRequest(
    const ChatRequest& req,
    ChatResponse& resp)
{
    // 1. Detect domain and switch adapter
    std::string domain = classifyDomain(req.message);
    runtime.switchLoraWeights(domain + "_adapter", stream);

    // 2. Check system prompt cache
    std::string systemPrompt = getSystemPrompt(domain);
    runtime.loadSystemPromptCache(systemPrompt, domain + "_adapter", stream);

    // 3. Generate with EAGLE (2-3x faster!)
    LLMGenerationRequest genReq;
    genReq.systemPrompt = systemPrompt;
    genReq.inputText = req.message;
    genReq.maxLength = 200;

    LLMGenerationResponse genResp;
    specDecodeRuntime.handleRequest(genReq, genResp, stream);

    resp.reply = genResp.outputText;
}
```

---

## Feynman Self-Test

After completing this level, you should be able to answer:

- [ ] **How does EAGLE achieve 2-3x speedup?**
  > Draft model generates a tree of candidates, base model verifies all at once, accepting multiple tokens per forward pass

- [ ] **Why can LoRA adapters switch instantly?**
  > They're just small matrices bound as graph inputs - switching means updating tensor pointers, not reloading the model

- [ ] **How do VLMs process images?**
  > Vision encoder converts image to embeddings, which replace <image> tokens in the text sequence before LLM processing

- [ ] **Why doesn't EAGLE work with CUDA graphs?**
  > CUDA graphs record fixed tensor shapes, but EAGLE tree structure varies with acceptance rate

- [ ] **What determines LoRA adapter size?**
  > rank × hidden_size × 2 (A and B matrices) × number of adapted layers

---

## Quick Commands Reference

```bash
# Export model with EAGLE support
tensorrt-edgellm-export-llm --model_dir ... --eagle_base

# Export draft model
tensorrt-edgellm-export-draft --base_model_dir ... --draft_model_dir ...

# Build EAGLE engines (both base and draft)
./llm_build --onnxDir base_onnx --engineDir base_engine --eagle_base
./llm_build --onnxDir draft_onnx --engineDir draft_engine --eagle_draft

# Run inference with EAGLE
./llm_inference_specdecode \
    --baseEngineDir base_engine \
    --draftEngineDir draft_engine \
    --inputFile input.json

# Export with LoRA support
tensorrt-edgellm-export-llm --max_lora_rank 64

# Run with LoRA adapters
./llm_inference \
    --engineDir engine \
    --loraWeights "medical:medical.safetensors,legal:legal.safetensors"

# Export VLM
tensorrt-edgellm-export-visual --model_dir Qwen/Qwen2-VL-7B

# Run VLM inference
./llm_inference \
    --engineDir llm_engine \
    --multimodalEngineDir vision_engine \
    --inputFile input_with_images.json
```

---

## What's Next?

You've mastered the advanced features! You can now:
- ✅ Deploy EAGLE for faster generation
- ✅ Manage multiple LoRA adapters at runtime
- ✅ Process images with VLM support
- ✅ Combine features for complex applications

**Next Level**: [08 Quick Reference](08_quick_reference.md) - Command cheatsheet and configuration reference.

---

*← [06 Optimizations](06_optimizations.md) | [08 Quick Reference →](08_quick_reference.md)*
