# TensorRT-Edge-LLM: Complete Learning Path for Contributors

**Goal**: Go from beginner to contributor in ~10 hours with hands-on exercises.
**Interview Ready**: Know every optimization for Jetson Orin/Thor AGX inside and out.

---

## Hour 1: First Contact (Covered in previous guide)
- Install, quantize, export, build engine, run inference
- See `/home/ajiang2/.claude/plans/staged-booping-wren.md`

---

## Hour 2: Understanding the Python Pipeline (Model Side)

### Concept: How Does the Code Know Which Model to Use?

**Simple answer**: It reads a config file and checks the `model_type` field.

```python
# The magic happens here:
# File: tensorrt_edgellm/llm_models/model_utils.py

def _check_model_type(model_dir: str, model_identifier: str) -> bool:
    """Check if model matches an identifier like 'llama' or 'qwen'"""
    cfg = AutoConfig.from_pretrained(model_dir)

    # Check 1: Does model_type contain our identifier?
    model_type = str(getattr(cfg, "model_type", "")).lower()
    if model_identifier in model_type:
        return True

    # Check 2: Does architectures list contain it?
    archs = getattr(cfg, "architectures", []) or []
    return any(model_identifier in str(a).lower() for a in archs)
```

**Try it yourself** - Create a file `learn_model_detection.py`:
```python
"""Exercise: Understand how model detection works"""
from transformers import AutoConfig

def detect_model_type(model_name: str):
    """Load a model's config and print what we find"""
    print(f"\n=== Checking: {model_name} ===")

    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # What the code looks at:
    print(f"model_type: {getattr(cfg, 'model_type', 'NOT FOUND')}")
    print(f"architectures: {getattr(cfg, 'architectures', 'NOT FOUND')}")

    # Key numbers for the model:
    print(f"hidden_size: {getattr(cfg, 'hidden_size', '?')}")
    print(f"num_hidden_layers: {getattr(cfg, 'num_hidden_layers', '?')}")
    print(f"num_attention_heads: {getattr(cfg, 'num_attention_heads', '?')}")
    print(f"vocab_size: {getattr(cfg, 'vocab_size', '?')}")

# Run on different models to see the pattern:
detect_model_type("Qwen/Qwen2.5-0.5B-Instruct")
detect_model_type("meta-llama/Llama-3.2-1B")
# detect_model_type("your-model-here")
```

**What you'll learn**: Every model has a config.json with these fields. The code is model-agnostic - it just reads these numbers.

---

### Concept: What Does Quantization Actually Do?

**Simple answer**: It shrinks numbers from 16 bits to 8 or 4 bits.

Think of it like this:
- FP16: "The temperature is 72.456789 degrees" (precise but takes space)
- FP8: "The temperature is 72.5 degrees" (good enough, half the space)
- INT4: "The temperature is about 70" (rough but tiny)

**Try it yourself** - Create `learn_quantization.py`:
```python
"""Exercise: See what quantization does to weights"""
import torch
from transformers import AutoModelForCausalLM

def show_weight_stats(model, layer_name="model.layers.0.self_attn.q_proj.weight"):
    """Show statistics of a weight tensor"""
    # Navigate to the weight
    parts = layer_name.split(".")
    obj = model
    for part in parts:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)

    weight = obj.data
    print(f"\nWeight: {layer_name}")
    print(f"  Shape: {weight.shape}")
    print(f"  Dtype: {weight.dtype}")
    print(f"  Min: {weight.min().item():.6f}")
    print(f"  Max: {weight.max().item():.6f}")
    print(f"  Mean: {weight.mean().item():.6f}")
    print(f"  Memory: {weight.numel() * weight.element_size() / 1024:.2f} KB")

# Load a tiny model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="cpu"
)

# Look at one attention weight
show_weight_stats(model)

# After quantization, these numbers get "binned" into fewer possible values
# FP16 has 65,536 possible values
# FP8 has 256 possible values
# INT4 has only 16 possible values!
```

---

### Concept: What Does ONNX Export Do?

**Simple answer**: It converts the model from "Python code + weights" to "a graph file any runtime can read".

Think of it like saving a recipe:
- PyTorch model = Chef who knows the recipe by heart
- ONNX file = Written recipe anyone can follow

**Try it yourself** - Create `learn_onnx_export.py`:
```python
"""Exercise: Export a tiny model to ONNX and inspect it"""
import torch
import torch.nn as nn

# 1. Create a simple "model" (just 2 layers)
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 5)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return self.layer2(x)

model = TinyModel()

# 2. Create example input
example_input = torch.randn(1, 10)

# 3. Export to ONNX
torch.onnx.export(
    model,
    example_input,
    "tiny_model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)
print("Exported to tiny_model.onnx")

# 4. Inspect the ONNX file
import onnx
onnx_model = onnx.load("tiny_model.onnx")

print("\n=== ONNX Graph ===")
print(f"Inputs: {[i.name for i in onnx_model.graph.input]}")
print(f"Outputs: {[o.name for o in onnx_model.graph.output]}")
print(f"Nodes (operations):")
for node in onnx_model.graph.node:
    print(f"  {node.op_type}: {node.input} -> {node.output}")
```

**What you'll see**: The ONNX file is just a list of operations (MatMul, Relu, etc.) with their connections. TensorRT reads this and optimizes it.

---

## Hour 3: Understanding the C++ Runtime (Inference Side)

### Concept: What is a TensorRT Plugin?

**Simple answer**: A custom operation that runs fast on GPU.

Think of it like this:
- TensorRT has built-in operations (add, multiply, etc.)
- But attention in LLMs is special and complex
- A plugin lets you write custom GPU code for attention
- TensorRT calls your plugin at the right time

**Read this file first**: `cpp/plugins/attentionPlugin/attentionPlugin.h`

The key interface looks like:
```cpp
class AttentionPlugin : public nvinfer1::IPluginV2DynamicExt {
    // TensorRT asks: "What size is your output?"
    DimsExprs getOutputDimensions(...);

    // TensorRT asks: "How much temp memory do you need?"
    size_t getWorkspaceSize(...);

    // TensorRT says: "Run your computation now!"
    int32_t enqueue(...);  // This is where the action happens
};
```

---

### Concept: What is a CUDA Kernel?

**Simple answer**: A function that runs on thousands of GPU threads at once.

Think of it like this:
- CPU: One chef cooking one dish at a time
- GPU: 1000 chefs each cooking one ingredient simultaneously

**Try it yourself** - Read the simplest kernel in the codebase:

File: `cpp/kernels/embeddingKernels/embeddingKernels.cu`

```cpp
// This kernel looks up embeddings for tokens
// Each thread handles a small piece of one embedding

__global__ void embeddingLookupKernel(
    int32_t const* inputIds,        // Token IDs like [5, 102, 7, ...]
    half const* embeddingTable,     // Big table of vectors
    half* output,                   // Where to write results
    int64_t batchSize,
    int64_t seqLen,
    int64_t hiddenSize)
{
    // "Which token am I handling?"
    // (Each thread figures this out from its ID)
    uint32_t tokenIdx = blockIdx.x * blockDim.y + threadIdx.y;

    // "What token ID is at this position?"
    int32_t tokenId = inputIds[tokenIdx];

    // "Copy that token's embedding to output"
    // (Each thread copies a few numbers, they work together)
    for (int i = threadIdx.x; i < hiddenSize; i += 32) {
        output[tokenIdx * hiddenSize + i] =
            embeddingTable[tokenId * hiddenSize + i];
    }
}
```

**The pattern**:
1. Each thread asks "which piece of work am I doing?" (using blockIdx, threadIdx)
2. Each thread does its small piece
3. All threads run simultaneously

---

### Exercise: Trace an Inference Call

**Goal**: Understand the code path from "input text" to "output text"

Create `trace_inference.md` and fill in as you read the code:

```markdown
# Inference Trace

## Step 1: Input JSON is parsed
File: examples/llm/llm_inference.cpp
Function: parseInputRequest()
What happens: JSON -> LLMGenerationRequest struct

## Step 2: Request goes to runtime
File: cpp/runtime/llmInferenceRuntime.cpp
Function: handleRequest()
What happens:
  - Tokenize input text
  - Run prefill (process all input tokens at once)
  - Run decode (generate one token at a time)

## Step 3: Prefill phase
Function: runPrefill()
What happens:
  - Embedding lookup (token IDs -> vectors)
  - Run through all transformer layers
  - Store KV cache for later

## Step 4: Decode phase (loop)
Function: runDecode()
What happens (repeated until done):
  - Take last token
  - Run through layers (using cached KV)
  - Sample next token
  - Check if we hit end token

## Step 5: Output
What happens:
  - Detokenize (token IDs -> text)
  - Write to output JSON
```

**Files to read** (in order):
1. `examples/llm/llm_inference.cpp` - Entry point
2. `cpp/runtime/llmInferenceRuntime.h` - Main class
3. `cpp/runtime/llmEngineRunner.h` - Engine execution

---

## Hour 4: Making Your First Contribution

### Setup: Get Ready to Contribute

```bash
# 1. Fork the repo on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM

# 3. Add upstream remote
git remote add upstream https://github.com/NVIDIA/TensorRT-Edge-LLM.git

# 4. Install pre-commit hooks (REQUIRED)
pip install pre-commit
pre-commit install

# 5. Create a branch for your work
git checkout -b my-first-contribution
```

### Exercise: Fix a Documentation Issue

**Goal**: Make a safe first contribution

1. Find something in the docs that could be clearer
2. Edit the file in `docs/source/developer_guide/`
3. Run pre-commit:
   ```bash
   pre-commit run --all-files
   ```
4. Commit with sign-off:
   ```bash
   git commit -s -m "docs: Clarify X in Y guide"
   ```

### Exercise: Add a Simple Test Case

**Goal**: Understand the test system

Create a new test input file `tests/test_cases/llm_custom.json`:
```json
{
    "batch_size": 1,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "max_generate_length": 50,
    "requests": [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "Write a Python hello world."}
            ]
        }
    ]
}
```

Run it:
```bash
./build/examples/llm/llm_inference \
    --engineDir ./engines/qwen3-0.6b \
    --inputFile tests/test_cases/llm_custom.json \
    --outputFile output.json
```

---

## Hour 5: Deep Dive - Choose Your Path

### Path A: Add Support for a New Model (Python focus)

**When you'd do this**: A new model came out that's similar to existing ones.

**Steps**:
1. Check if it works automatically:
   ```python
   from tensorrt_edgellm import quantize_and_save_llm
   quantize_and_save_llm(
       model_dir="new-model-name",
       output_dir="./test_output",
       quantization="fp8"
   )
   ```

2. If it fails, check the error. Common issues:
   - Missing chat template → Add to `tensorrt_edgellm/chat_templates/`
   - Special architecture → May need changes to `llm_models/layers/layers.py`

3. Add test configuration to `tests/test_lists/l0_pipeline_a30.yml`:
   ```yaml
   - tests/defs/test_model_export.py::test_model_export[NewModel-fp16-mxsl4096-mxbs1-mxil2048]
   ```

**Key files to understand**:
- `tensorrt_edgellm/llm_models/model_utils.py` - Model detection
- `tensorrt_edgellm/llm_models/models/llm_model.py` - Model wrapper
- `tensorrt_edgellm/chat_templates/` - Chat formatting

---

### Path B: Optimize a CUDA Kernel (C++ focus)

**When you'd do this**: You found a performance bottleneck.

**Steps**:
1. Profile to find the slow part:
   ```bash
   ./build/examples/llm/llm_inference \
       --engineDir ./engines/model \
       --inputFile input.json \
       --dumpProfile \
       --profileOutputFile profile.json
   ```

2. Look at the kernel code in `cpp/kernels/`

3. Common optimizations:
   - Increase work per thread (fewer kernel launches)
   - Use shared memory (faster than global memory)
   - Coalesce memory access (adjacent threads read adjacent memory)

**Key files to understand**:
- `cpp/kernels/embeddingKernels/` - Simplest kernels
- `cpp/kernels/decodeAttentionKernels/` - Attention decode
- `cpp/plugins/attentionPlugin/` - Main attention plugin

---

### Path C: Add a New Feature (Full stack)

**Example**: Add support for a new sampling method

**Steps**:
1. Understand current sampling in `cpp/sampler/`
2. Add the new method
3. Expose it in the input JSON schema
4. Update `cpp/runtime/llmRuntimeUtils.h` for the new parameter
5. Add tests

**Key files**:
- `cpp/sampler/samplingKernels.cu` - Sampling implementations
- `cpp/runtime/llmRuntimeUtils.h` - Request/response structures
- `examples/llm/INPUT_FORMAT.md` - Input documentation

---

## Quick Reference: Contribution Checklist

Before submitting a PR:

- [ ] Pre-commit passes: `pre-commit run --all-files`
- [ ] C++ builds: `cd build && make -j$(nproc)`
- [ ] Python package builds: `python -m build --wheel`
- [ ] Tests pass: `pytest --priority=l0_pipeline_a30 -v`
- [ ] Commit is signed: `git commit -s`
- [ ] Commit message follows format: `feat: Add X` or `fix: Resolve Y`

---

## Code Style Cheat Sheet

### Python
```python
# Good: Type hints, docstrings, 120 char max
def process_model(model_dir: str, output_dir: str) -> bool:
    """Process a model and save results.

    Args:
        model_dir: Path to input model
        output_dir: Path for output

    Returns:
        True if successful
    """
    pass
```

### C++
```cpp
// Good: camelCase functions, PascalCase classes, 4-space indent
class MyProcessor
{
public:
    bool processInput(int32_t inputSize)
    {
        // Always use braces, even for one-liners
        if (inputSize > 0)
        {
            return true;
        }
        return false;
    }

private:
    int32_t mBufferSize;  // m prefix for members
};
```

---

## Next Steps After 5 Hours

1. **Join the community**: Watch the GitHub repo for issues labeled "good first issue"
2. **Read more code**: Pick one kernel and understand it deeply
3. **Run benchmarks**: Use `--dumpProfile` to understand performance
4. **Try VLMs**: The visual encoder adds another dimension
5. **Try EAGLE**: Speculative decoding is an advanced optimization

---

## Files This Guide References

| Purpose | File |
|---------|------|
| Model detection | `tensorrt_edgellm/llm_models/model_utils.py` |
| Model wrapper | `tensorrt_edgellm/llm_models/models/llm_model.py` |
| Quantization | `tensorrt_edgellm/quantization/llm_quantization.py` |
| ONNX export | `tensorrt_edgellm/onnx_export/llm_export.py` |
| C++ runtime | `cpp/runtime/llmInferenceRuntime.h` |
| Attention plugin | `cpp/plugins/attentionPlugin/attentionPlugin.h` |
| Embedding kernel | `cpp/kernels/embeddingKernels/embeddingKernels.cu` |
| Test config | `tests/test_lists/l0_pipeline_a30.yml` |
| Contribution guide | `CONTRIBUTING.md` |
| Style guide | `CODING_GUIDELINES.md` |

---

## Hour 6: C++ Deep Dive - Memory & Tensors

### Concept: The Tensor Class

**What it is**: A box that holds numbers on GPU or CPU.

**Two ownership modes**:
```cpp
// Mode 1: Tensor OWNS memory (allocates, frees automatically)
Tensor t1({1024, 768}, DeviceType::kGPU, DataType::kFLOAT);
// Memory allocated on GPU, freed when t1 dies

// Mode 2: Tensor BORROWS memory (just points to existing data)
void* existingGpuMemory = somePointer;
Tensor t2(existingGpuMemory, {1024, 768}, DeviceType::kGPU, DataType::kFLOAT);
// No allocation, no freeing - just a view
```

### Exercise 6.1: Trace Memory Ownership

**File**: `cpp/common/tensor.h`

Create `exercises/ex6_tensor_trace.cpp`:
```cpp
// Exercise: Understand tensor ownership by reading the code

#include <iostream>

// TASK 1: Find these in tensor.h and explain what they do
// - mData (what is it?)
// - mOwnMemory (what does this flag mean?)
// - DeviceType enum (what are the options?)

// TASK 2: Answer these questions by reading the code:
// Q1: What happens when you copy a Tensor?
// Q2: What happens when you move a Tensor?
// Q3: How does the destructor know whether to free memory?

// TASK 3: Find where cudaMalloc is called
// Hint: Look for allocateMemory() function

int main() {
    std::cout << "Read cpp/common/tensor.h and answer the questions above\n";
    return 0;
}
```

### Exercise 6.2: KV Cache Memory Layout

**What it is**: A big GPU buffer storing all the Key and Value vectors from previous tokens.

**Shape**: `[numLayers, batchSize, 2, numKVHeads, maxSeqLen, headDim]`

```
Example for a 32-layer model with batch=4:
[32 layers] x [4 sequences] x [K and V] x [8 heads] x [4096 max tokens] x [128 dim]
= 32 * 4 * 2 * 8 * 4096 * 128 = ~8.5 billion elements
= ~17GB in FP16!
```

**File**: `cpp/runtime/linearKVCache.h`

**Questions to answer**:
1. Where is the cache memory allocated? (find cudaMalloc)
2. What does `mDeviceKVCacheLengths` store?
3. What happens when you call `resetKVCache()`?
4. Why is it called "Linear" KV Cache?

---

## Hour 7: C++ Deep Dive - The Decode Loop

### Concept: Prefill vs Decode

**Prefill**: Process ALL input tokens at once
- Input: 512 tokens → Output: Logits for token 513
- KV Cache: Filled with 512 entries

**Decode**: Process ONE token at a time
- Input: 1 token → Output: Logits for next token
- KV Cache: Add 1 entry

### Exercise 7.1: Trace the Prefill Path

**File**: `cpp/runtime/llmEngineRunner.cpp`

Find `executePrefillStep()` and answer:
1. What shape is inputIds?
2. What gets bound to the TensorRT engine?
3. Where does actual execution happen? (look for enqueueV3)
4. What happens to KV cache after prefill?

### Exercise 7.2: Trace the Decode Path

**File**: `cpp/runtime/llmEngineRunner.cpp`

Find `executeVanillaDecodingStep()` and answer:
1. Why is input shape [batch, 1] instead of [batch, seqLen]?
2. What is mSequenceContextLengths and why does it increment?
3. Where is the CUDA graph launched (if captured)?

### Exercise 7.3: Build a Mental Model

Fill in the blanks:

```
After prefill with 100 tokens (batch=4):
- KV Cache lengths: [100, 100, 100, 100]
- Next input shape: [4, 1]

After 10 decode steps:
- KV Cache lengths: [110, 110, 110, 110]
- Total tokens generated: 10 per sequence
```

---

## Hour 8: Edge Optimizations for Jetson Orin/Thor

### Optimization 1: CUDA Graphs

**What it is**: Record GPU commands once, replay instantly.

**Why it matters on Jetson**: CPU is slower than desktop, so reducing CPU overhead helps.

**File**: `cpp/runtime/llmEngineRunner.h` (lines 145-155, 198-204)

```cpp
// Storage: mCudaGraphs maps config hash → captured graph
std::unordered_map<size_t, std::pair<cudaGraph_t, cudaGraphExec_t>> mCudaGraphs;
```

**Exercise**: Find `captureVanillaDecodingCudaGraph()` and trace:
- What triggers graph capture?
- What's the "hash" used as key?
- Where does `cudaGraphLaunch` get called?

### Optimization 2: Memory Pre-allocation

**What it is**: Allocate all GPU memory once at startup, never again.

**Why it matters**: cudaMalloc is slow (~1ms). During 100-token generation, that's 100ms wasted.

**File**: `cpp/runtime/llmEngineRunner.cpp`

**Exercise**: Find where `mExecContextMemory` is allocated:
- What's its size?
- How is it shared between prefill and generation contexts?

### Optimization 3: Linear KV Cache (vs Paged)

**What it is**: Simple contiguous memory layout.

**Why it matters on Jetson**:
- Simpler = faster memory access patterns
- No page table overhead
- Predictable memory usage

```cpp
// Linear: [layer0_seq0_K][layer0_seq0_V][layer0_seq1_K]...
// Paged:  [page_ptr][page_ptr] → scattered memory locations
```

---

## Hour 9: Advanced Features Deep Dive

### Feature 1: EAGLE Speculative Decoding

**What it is**: Use a small "draft" model to guess multiple tokens, verify with the big model.

**Why it's faster**:
- Draft model is tiny (runs fast)
- Verify multiple guesses in ONE forward pass
- Accept 2-5 tokens per iteration instead of 1

**Files**:
- `cpp/runtime/llmInferenceSpecDecodeRuntime.h`
- `cpp/kernels/speculative/eagleAcceptKernels.cu`

**Exercise**: Trace the EAGLE flow:
1. Prefill: Base model processes input, draft model gets hidden states
2. Draft: Generate `draftingStep` tokens as a tree
3. Verify: Base model evaluates tree, accept matching tokens

**Configuration**:
```cpp
struct EagleDraftingConfig {
    int32_t draftingTopK;   // Branches per tree node
    int32_t draftingStep;   // Tree depth
    int32_t verifyTreeSize; // Tokens to verify at once
};
```

### Feature 2: LoRA Runtime Switching

**What it is**: Swap fine-tuned adapters without reloading the model.

**File**: `cpp/runtime/llmEngineRunner.h` (lines 157-173)

```cpp
std::unordered_map<std::string, std::vector<rt::Tensor>> mLoraWeights{};
std::string mActiveLoraWeightsName{};
```

**Exercise**: Answer:
1. What does `addLoraWeights()` do?
2. What does `switchLoraWeights()` do?
3. Why separate CUDA graphs per LoRA adapter?

### Feature 3: System Prompt KV Cache

**What it is**: Cache KV values for common system prompts.

**File**: `cpp/runtime/llmInferenceRuntime.h` (lines 35-42)

```cpp
struct SystemPromptKVCache {
    std::string systemPrompt;
    std::vector<tokenizer::Rank> tokenizedPrompt;
    rt::Tensor kvCacheContent;  // The cached KV!
};
```

**Performance benefit**:
- System prompt: 500 tokens, User prompt: 50 tokens
- Without caching: Prefill 550 tokens
- With caching: Prefill 50 tokens → 11x faster prefill!

---

## Hour 10: Interview Preparation

### The Optimization Checklist

**Memory Optimizations**:
- [ ] Pre-allocated memory (no runtime malloc)
- [ ] Linear KV cache (contiguous memory)
- [ ] Reduced vocabulary (smaller logits)
- [ ] FP16/FP8/INT4 quantization

**Compute Optimizations**:
- [ ] CUDA graphs (reduce CPU overhead)
- [ ] Fused attention kernels (no intermediate writes)
- [ ] EAGLE speculative decoding (more tokens per step)
- [ ] System prompt caching (skip repeated prefill)

**Edge-Specific**:
- [ ] Single-batch optimized (no continuous batching overhead)
- [ ] Synchronous execution model (simpler scheduling)
- [ ] TensorRT engine optimization (layer fusion, precision)

### Exercise 10.1: Explain Each Optimization

For each, write a 2-3 sentence explanation a non-expert could understand.

### Exercise 10.2: Code Walkthrough Practice

Start at: `examples/llm/llm_inference.cpp main()`
End at: First token generated

Walk through and time yourself - can you explain in 10 minutes?

### Exercise 10.3: Debug Scenarios

**Scenario 1: Slow First Token**
"First token takes 5 seconds, subsequent tokens are 50ms each"
- Check: CUDA graph captured? Prefill too long? Model loading included?

**Scenario 2: Out of Memory**
"CUDA OOM on Jetson Orin 32GB"
- Check: maxKVCacheCapacity? Batch size? Model size?

**Scenario 3: Wrong Outputs**
"Model generates garbage text"
- Check: Tokenizer mismatch? Chat template? Sampling params?

---

## Quick Interview Answers

**Q: Why TensorRT for edge LLM?**
A: TensorRT optimizes the compute graph (fuses layers, picks fastest kernels, uses tensor cores). On Jetson, every FLOP matters.

**Q: Why not use vLLM/TGI on Jetson?**
A: They're designed for server GPUs with continuous batching. Edge inference is single-user, so simpler synchronous execution is better.

**Q: What's the biggest memory consumer?**
A: KV Cache. For a 7B model with 4K context, it's ~14GB. That's why maxKVCacheCapacity is critical.

**Q: How does EAGLE help latency?**
A: Instead of 1 token per forward pass, we verify a tree of ~10 candidates and accept ~3. That's 3x tokens per GPU round-trip.

**Q: Why pre-allocate everything?**
A: cudaMalloc is slow (~1ms). During 100-token generation, that's 100ms wasted. Pre-allocation = zero allocation during inference.

---

## Extended Files Reference (Hours 6-10)

| Topic | File |
|-------|------|
| Tensor class | `cpp/common/tensor.h` |
| KV Cache | `cpp/runtime/linearKVCache.h` |
| Engine Runner | `cpp/runtime/llmEngineRunner.h` |
| CUDA Graphs | `cpp/runtime/llmEngineRunner.cpp` |
| EAGLE | `cpp/runtime/llmInferenceSpecDecodeRuntime.h` |
| LoRA | `cpp/runtime/llmEngineRunner.h:157-173` |
| System Prompt Cache | `cpp/runtime/llmInferenceRuntime.h:35-42` |
| Attention Kernels | `cpp/kernels/contextAttentionKernels/` |
| Decode Attention | `cpp/kernels/decodeAttentionKernels/` |
| Metrics | `cpp/profiling/metrics.h` |
