# Level 1: Getting Started

**Reading Time: 30 minutes**

Imagine you have a very smart friend who can answer any question, but they're a slow talker. LLM inference acceleration is like teaching that friend to use shorthand notes and predict what you'll ask next - they give the same great answers, just much faster. TensorRT-Edge-LLM is the tool that makes this happen on NVIDIA's edge devices.

## The Core Insight First

1. **TensorRT-Edge-LLM converts HuggingFace models into fast, edge-optimized engines**
2. **The pipeline is: Python (prepare) → C++ (build) → C++ (run)**
3. **Everything is pre-computed and pre-allocated - zero runtime overhead**

## Key Numbers to Memorize

| Metric | Value | Context |
|--------|-------|---------|
| **Smallest test model** | Qwen3-0.6B | Use for learning, ~1.2GB |
| **Typical edge model** | 3-7B params | Fits on Jetson Orin 32GB |
| **Prefill throughput** | ~2000 tokens/sec | Processing input |
| **Decode throughput** | 20-50 tokens/sec | Generating output |
| **First token latency** | 100-500ms | Depends on prompt length |

## Table of Contents

1. [Understanding the Pipeline](#1-understanding-the-pipeline) (5 min)
2. [Your First Inference](#2-your-first-inference) (15 min)
3. [Project Structure](#3-project-structure) (10 min)

---

## 1. Understanding the Pipeline

### The Three Stages

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        THE THREE-STAGE PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STAGE 1: EXPORT (Python, on x86 host)                                      │
│  ═══════════════════════════════════════                                    │
│                                                                             │
│     HuggingFace Model                                                       │
│           │                                                                 │
│           ▼                                                                 │
│     ┌─────────────┐      ┌──────────────┐      ┌────────────────┐          │
│     │  Quantize   │ ───► │ Export ONNX  │ ───► │ Config + ONNX  │          │
│     │  (FP8/INT4) │      │              │      │    files       │          │
│     └─────────────┘      └──────────────┘      └────────────────┘          │
│                                                                             │
│  STAGE 2: BUILD (C++, on Jetson device)                                     │
│  ══════════════════════════════════════                                     │
│                                                                             │
│     ONNX files                                                              │
│           │                                                                 │
│           ▼                                                                 │
│     ┌─────────────────────┐      ┌────────────────────────┐                │
│     │  TensorRT Compiler  │ ───► │  Optimized Engine      │                │
│     │  (layer fusion,     │      │  (.plan file)          │                │
│     │   kernel selection) │      │                        │                │
│     └─────────────────────┘      └────────────────────────┘                │
│                                                                             │
│  STAGE 3: RUN (C++, on Jetson device)                                       │
│  ════════════════════════════════════                                       │
│                                                                             │
│     Engine + Input                                                          │
│           │                                                                 │
│           ▼                                                                 │
│     ┌─────────────┐      ┌──────────────┐      ┌────────────────┐          │
│     │  Tokenize   │ ───► │   Prefill    │ ───► │    Decode      │ ──► Text │
│     │             │      │   (batch)    │      │    (loop)      │          │
│     └─────────────┘      └──────────────┘      └────────────────┘          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Three Stages?

| Stage | Where It Runs | Why Separate |
|-------|---------------|--------------|
| **Export** | Powerful x86 server | Quantization needs lots of memory and compute |
| **Build** | Target device | Engine is optimized for specific GPU architecture |
| **Run** | Target device | Inference happens where you need it |

### Mental Model: Compiling a Book

Think of it like publishing a book:
- **Export** = Write the manuscript (content creation)
- **Build** = Print the book (format for the medium)
- **Run** = Read the book (use the final product)

You write once, print for each format (paperback, kindle), and read many times.

---

## 2. Your First Inference

### Prerequisite Check

```bash
# Check Python environment
python --version  # Should be 3.8+

# Check CUDA (if on GPU machine)
nvidia-smi  # Should show GPU info

# Check TensorRT-Edge-LLM installation
python -c "import tensorrt_edgellm; print('OK')"
```

### Step 1: Quantize a Model (5 min)

```bash
# Using the smallest model for quick testing
tensorrt-edgellm-quantize-llm \
    --model_dir Qwen/Qwen2.5-0.5B-Instruct \
    --output_dir ./my_first_model/quantized \
    --quantization fp8
```

**What just happened?**
- Downloaded model from HuggingFace (if not cached)
- Converted weights from FP16 (16-bit) to FP8 (8-bit)
- Saved quantized weights to `./my_first_model/quantized/`

### Step 2: Export to ONNX (3 min)

```bash
tensorrt-edgellm-export-llm \
    --model_dir ./my_first_model/quantized \
    --output_dir ./my_first_model/onnx
```

**What just happened?**
- Traced the PyTorch model execution
- Saved as ONNX graph (portable neural network format)
- Applied "graph surgery" (optimizations for TensorRT)

### Step 3: Build TensorRT Engine (5 min)

```bash
# This step requires a Jetson device or compatible GPU
./build/examples/llm/llm_build \
    --onnxDir ./my_first_model/onnx \
    --engineDir ./my_first_model/engine \
    --maxBatchSize 1 \
    --maxInputLen 512 \
    --maxKVCacheCapacity 2048
```

**What just happened?**
- TensorRT analyzed the ONNX graph
- Fused operations for better performance
- Selected optimal CUDA kernels for this GPU
- Saved optimized engine to `./my_first_model/engine/`

### Step 4: Run Inference (2 min)

Create `input.json`:
```json
{
    "batch_size": 1,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "max_generate_length": 100,
    "requests": [
        {
            "messages": [
                {"role": "user", "content": "What is machine learning in one sentence?"}
            ]
        }
    ]
}
```

Run inference:
```bash
./build/examples/llm/llm_inference \
    --engineDir ./my_first_model/engine \
    --inputFile input.json \
    --outputFile output.json

# View the response
cat output.json | python -m json.tool
```

### What You Just Did

```
Qwen2.5-0.5B (HuggingFace)
         │
         ▼ quantize (FP16 → FP8)
         │
    Quantized Model
         │
         ▼ export (PyTorch → ONNX)
         │
     ONNX Files
         │
         ▼ build (ONNX → TensorRT)
         │
    TensorRT Engine
         │
         ▼ inference
         │
    "Machine learning is..."
```

### Hands-on Exercise

Try modifying the inference:

1. **Change the prompt** in `input.json`
2. **Adjust temperature** (0.1 = deterministic, 1.0 = creative)
3. **Increase max_generate_length** to 200

```bash
# Re-run with your changes
./build/examples/llm/llm_inference \
    --engineDir ./my_first_model/engine \
    --inputFile input.json \
    --outputFile output.json
```

---

## 3. Project Structure

### Directory Map

```
TensorRT-Edge-LLM/
├── tensorrt_edgellm/          # Python package (Stage 1: Export)
│   ├── scripts/               # CLI entry points
│   │   ├── quantize_llm.py   # tensorrt-edgellm-quantize-llm
│   │   └── export_llm.py     # tensorrt-edgellm-export-llm
│   ├── llm_models/           # Model implementations
│   ├── quantization/         # Quantization logic
│   └── onnx_export/          # ONNX conversion
│
├── cpp/                       # C++ runtime (Stages 2 & 3)
│   ├── builder/              # Stage 2: Engine building
│   ├── runtime/              # Stage 3: Inference execution
│   ├── kernels/              # CUDA kernels
│   └── plugins/              # TensorRT plugins
│
├── examples/                  # Reference implementations
│   └── llm/
│       ├── llm_build.cpp     # Engine builder example
│       └── llm_inference.cpp # Inference example
│
└── tests/                     # Test suite
    └── test_cases/           # Sample inputs
```

### Key Files to Know

| File | Purpose | When You'd Read It |
|------|---------|-------------------|
| `tensorrt_edgellm/__init__.py` | Python public API | Using Python directly |
| `cpp/runtime/llmInferenceRuntime.h` | C++ inference API | Understanding runtime |
| `cpp/builder/builder.h` | Engine building API | Customizing builds |
| `examples/llm/llm_inference.cpp` | Full inference example | Learning patterns |

### The Flow of Data

```
                        YOUR INPUT
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│                      input.json                                   │
│  {                                                                │
│    "requests": [{"messages": [{"content": "Hello!"}]}]           │
│  }                                                                │
└───────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│                      Tokenizer                                    │
│  "Hello!" → [15496, 0]  (token IDs)                              │
└───────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│                      Prefill                                      │
│  Process all input tokens, fill KV cache                         │
│  Output: logits for next token                                   │
└───────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│                      Decode Loop                                  │
│  Repeat until done:                                              │
│    1. Sample next token from logits                              │
│    2. Run model on new token                                     │
│    3. Update KV cache                                            │
└───────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│                      Detokenizer                                  │
│  [15496, 0, 2822, 1073] → "Hello! How are"                       │
└───────────────────────────────────────────────────────────────────┘
                            │
                            ▼
                       YOUR OUTPUT
```

---

## Feynman Self-Test

After completing this level, you should be able to answer:

- [ ] **Can I explain the three-stage pipeline in one sentence?**
  > "Export converts models to ONNX, build compiles for the GPU, run executes inference."

- [ ] **What's the smallest model I can use for testing?**
  > Qwen2.5-0.5B-Instruct (~1.2GB)

- [ ] **What does quantization do?**
  > Shrinks weight precision (FP16 → FP8) to save memory while maintaining quality

- [ ] **Where does the KV cache live?**
  > GPU memory, pre-allocated during engine build

- [ ] **What's the difference between prefill and decode?**
  > Prefill processes all input at once; decode generates one token at a time

## If You're Stuck

### "Command not found: tensorrt-edgellm-*"
```bash
# Reinstall the Python package
pip install dist/*.whl --force-reinstall
```

### "CUDA out of memory"
```bash
# Use a smaller model or reduce settings
--maxKVCacheCapacity 1024  # Reduce from 2048
--maxInputLen 256          # Reduce from 512
```

### "Engine build fails"
```bash
# Check TensorRT installation
python -c "import tensorrt; print(tensorrt.__version__)"
# Verify CUDA version matches
nvcc --version
```

### Need More Help?
- Check `CONTRIBUTING.md` for debugging tips
- Look at `tests/test_cases/` for working examples
- Read error messages carefully - they're often descriptive

---

## What's Next?

You've completed Level 1! You can now:
- ✅ Run the full inference pipeline
- ✅ Understand what each stage does
- ✅ Navigate the project structure

**Next Level**: [02 Architecture Deep Dive](02_architecture.md) - Learn WHY each component exists and how data flows through the system.

---

*← [README](README.md) | [02 Architecture →](02_architecture.md)*
