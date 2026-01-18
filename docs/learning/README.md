# LLM Inference Acceleration: From Zero to Expert

**Imagine you're a postal worker.** Every day, thousands of letters arrive. You could read each one word-by-word (slow), or you could scan the key parts and sort them in parallel (fast). LLM inference acceleration is about making AI read and respond faster by being smarter about memory, computation, and prediction.

## The Core Insight First

1. **LLMs are memory-bound, not compute-bound** - The GPU spends more time waiting for data than doing math
2. **The KV Cache is everything** - 70% of memory goes to storing what the model has already "read"
3. **Every millisecond of CPU overhead is wasted** - On edge devices, the CPU is the bottleneck, not the GPU

## Key Numbers to Memorize

| Component | Value | Why It Matters |
|-----------|-------|----------------|
| **KV Cache per 7B model** | ~14GB @ 4K context | This limits how long your conversations can be |
| **cudaMalloc latency** | ~1ms | 100 token generation = 100ms wasted if not pre-allocated |
| **Decode step latency** | ~30-50ms on Orin | This is your tokens-per-second ceiling |
| **EAGLE speedup** | 2-3x | Speculative decoding generates multiple tokens per step |
| **FP16 → FP8** | 2x memory savings | Same quality, half the memory |
| **Prefill vs Decode ratio** | 10:1 compute | Prefill is parallel, decode is sequential |

## Learning Path Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    YOUR JOURNEY TO EXPERTISE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Level 1: Getting Started (30 min)                                  │
│  ════════════════════════════════                                   │
│  "I can run inference and understand the pipeline"                  │
│                                                                     │
│          ↓                                                          │
│                                                                     │
│  Level 2: Core Concepts (3 hours)                                   │
│  ═══════════════════════════════                                    │
│  "I understand WHY each component exists"                           │
│                                                                     │
│          ↓                                                          │
│                                                                     │
│  Level 3: Platform Deep Dives (4 hours)                             │
│  ══════════════════════════════════════                             │
│  "I can modify Python export and C++ runtime code"                  │
│                                                                     │
│          ↓                                                          │
│                                                                     │
│  Level 4: Advanced Topics (3 hours)                                 │
│  ═════════════════════════════════                                  │
│  "I understand every optimization technique"                        │
│                                                                     │
│          ↓                                                          │
│                                                                     │
│  Level 5: Expert Reference (ongoing)                                │
│  ═══════════════════════════════════                                │
│  "I can debug, extend, and contribute"                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Table of Contents

| # | Tutorial | Time | You Will Learn |
|---|----------|------|----------------|
| **Level 1: Getting Started** ||||
| 01 | [Getting Started](01_getting_started.md) | 30 min | Run your first inference, understand the pipeline |
| **Level 2: Core Concepts** ||||
| 02 | [Architecture Deep Dive](02_architecture.md) | 1.5 hr | The three-stage pipeline, data flow, component roles |
| 03 | [Memory Model](03_memory_model.md) | 1.5 hr | KV cache, tensor management, GPU memory layout |
| **Level 3: Platform Deep Dives** ||||
| 04 | [Python Export Pipeline](04_python_pipeline.md) | 2 hr | Quantization, ONNX export, model surgery |
| 05 | [C++ Runtime Internals](05_cpp_runtime.md) | 2 hr | Engine execution, decode loop, attention kernels |
| **Level 4: Advanced Topics** ||||
| 06 | [Optimization Techniques](06_optimizations.md) | 1.5 hr | CUDA graphs, memory pre-allocation, fused kernels |
| 07 | [Advanced Features](07_advanced_features.md) | 1.5 hr | EAGLE, LoRA, system prompt caching, VLMs |
| **Level 5: Reference Materials** ||||
| 08 | [Quick Reference Cards](08_quick_reference.md) | - | Cheat sheets, command references, config options |
| 09 | [Interview Prep & Self-Test](09_interview_prep.md) | 1 hr | Technical questions, debug scenarios, expertise proof |

**Total Learning Time: ~12 hours** to go from zero to expert

## Sample Projects

| Project | Difficulty | Description |
|---------|------------|-------------|
| [hello_inference](sample_projects/hello_inference/) | Beginner | Complete pipeline walkthrough |
| [multi_adapter](sample_projects/multi_adapter/) | Intermediate | LoRA adapter management |
| [benchmark_suite](sample_projects/benchmark_suite/) | Advanced | Profile and compare configurations |

## The Mental Model: A Restaurant Kitchen

Think of LLM inference as a restaurant kitchen:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     THE RESTAURANT KITCHEN MODEL                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  CUSTOMER ORDER          =  Input prompt                            │
│  (what they want)            "Tell me about Paris"                  │
│                                                                     │
│  PREP STATION            =  Prefill phase                           │
│  (prepare ingredients)       Process all input tokens at once       │
│                                                                     │
│  COOKING LINE            =  Decode phase                            │
│  (cook one dish at a time)   Generate one token at a time           │
│                                                                     │
│  PANTRY                  =  KV Cache                                │
│  (ingredients on hand)       Store what we've already computed      │
│                                                                     │
│  RECIPE BOOK             =  Model weights                           │
│  (how to cook)               The neural network parameters          │
│                                                                     │
│  SOUS CHEF               =  Draft model (EAGLE)                     │
│  (prepares ahead)            Guesses next tokens in advance         │
│                                                                     │
│  SPECIAL SAUCES          =  LoRA adapters                           │
│  (customize dishes)          Fine-tuned behaviors you can swap      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**The insight**: A fast kitchen doesn't cook faster - it eliminates waiting. Same with LLM inference: we don't make the GPU faster, we eliminate time the GPU spends waiting for data or instructions.

## How to Use This Guide

### If You Have 1 Hour
Read Level 1 only. You'll understand:
- What the system does
- How to run inference
- The basic pipeline

### If You Have 4 Hours
Read Levels 1-2. You'll understand:
- Complete architecture
- Memory model
- Why each component exists

### If You Have 8 Hours
Read Levels 1-4. You'll be able to:
- Modify the Python export pipeline
- Understand C++ runtime code
- Apply optimization techniques

### If You Have 12+ Hours
Complete all levels. You'll be able to:
- Debug complex issues
- Extend the system
- Contribute to the codebase
- Pass technical interviews

## Prerequisites

| Skill | Level Needed | If You're Missing It |
|-------|--------------|---------------------|
| Python | Intermediate | [Python Tutorial](https://docs.python.org/3/tutorial/) |
| C++ | Basic | Just read the code, patterns are explained |
| CUDA | Conceptual | Explained in tutorials, no coding required |
| PyTorch | Basic | Understanding of tensors and models |
| Transformers | Conceptual | Explained as needed |

## Before You Start

```bash
# Clone the repository
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM

# Install Python dependencies
pip install -r requirements.txt

# Build Python package
pip install build
python -m build --wheel --outdir dist .
pip install dist/*.whl

# Verify installation
python -c "from tensorrt_edgellm import quantize_and_save_llm; print('Ready!')"
```

## Feynman Self-Test: Are You Ready?

Before diving in, can you answer these? (It's OK if not - that's why you're here!)

- [ ] What's the difference between prefill and decode?
- [ ] Why is KV cache the biggest memory consumer?
- [ ] What problem does CUDA graphs solve?
- [ ] Why use FP8 instead of FP16?

**If you can't answer these yet, perfect!** Start with [Level 1: Getting Started](01_getting_started.md).

---

*Next: [01 Getting Started →](01_getting_started.md)*
