# Level 3a: Python Export Pipeline Deep Dive

**Reading Time: 2 hours**

Think of the Python pipeline as a master chef who takes a fancy restaurant recipe (HuggingFace model) and adapts it for a food truck kitchen (edge device). The chef keeps the same flavors (model quality) but uses smaller portions (quantization), writes it in universal notation (ONNX), and removes unnecessary steps (graph surgery). This tutorial teaches you to be that chef.

## The Core Insight First

1. **Quantization trades precision for memory** - FP8 uses half the memory of FP16 with minimal quality loss
2. **ONNX is the universal translator** - Converts PyTorch's dynamic execution to a static graph
3. **Graph surgery optimizes for TensorRT** - Restructures operations for GPU-friendly execution

## Key Numbers to Memorize

| Metric | Value | Context |
|--------|-------|---------|
| **Calibration samples** | 512 | Default for quantization accuracy |
| **FP16 → FP8 ratio** | 2x | Memory savings |
| **FP16 → INT4 ratio** | 4x | Maximum compression |
| **Export time** | 5-30 min | Depends on model size |
| **Quantization time** | 10-60 min | Depends on calibration samples |

## Table of Contents

1. [Quantization Deep Dive](#1-quantization-deep-dive) (40 min)
2. [ONNX Export Internals](#2-onnx-export-internals) (30 min)
3. [Graph Surgery](#3-graph-surgery) (30 min)
4. [Adding New Model Support](#4-adding-new-model-support) (20 min)

---

## 1. Quantization Deep Dive

### What Quantization Actually Does

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        QUANTIZATION: THE BIG PICTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ORIGINAL (FP16): 16 bits per weight                                        │
│  ════════════════════════════════════                                       │
│                                                                             │
│    Weight value: 0.123456789...                                             │
│    Stored as:    [sign][exponent: 5 bits][mantissa: 10 bits]               │
│    Precision:    ~3-4 decimal places                                        │
│    Memory:       2 bytes × 7B params = 14 GB                               │
│                                                                             │
│  QUANTIZED (FP8): 8 bits per weight                                         │
│  ═══════════════════════════════════                                        │
│                                                                             │
│    Weight value: 0.123 (rounded)                                            │
│    Stored as:    [sign][exponent: 4 bits][mantissa: 3 bits]                │
│    Precision:    ~1-2 decimal places                                        │
│    Memory:       1 byte × 7B params = 7 GB                                 │
│                                                                             │
│  THE TRADEOFF:                                                              │
│  ┌────────────────┬─────────────────┬───────────────────────┐              │
│  │ Format         │ Memory per 7B   │ Quality Impact        │              │
│  ├────────────────┼─────────────────┼───────────────────────┤              │
│  │ FP32           │ 28 GB           │ Baseline              │              │
│  │ FP16           │ 14 GB           │ Negligible loss       │              │
│  │ FP8            │ 7 GB            │ ~1% loss              │              │
│  │ INT4 (AWQ)     │ 3.5 GB          │ ~2-3% loss            │              │
│  └────────────────┴─────────────────┴───────────────────────┘              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### How Calibration Works

```python
# File: tensorrt_edgellm/quantization/llm_quantization.py

def quantize_llm(model, tokenizer, quantization, dataset_dir, ...):
    """
    The calibration process:
    1. Run model on sample data
    2. Record activation ranges
    3. Choose scale factors that minimize error
    """

    # Step 1: Load calibration data
    dataloader = get_llm_calib_dataloader(
        tokenizer=tokenizer,
        dataset_dir=dataset_dir,  # e.g., "cnn_dailymail"
        batch_size=1,
        num_samples=512  # 512 examples for calibration
    )

    # Step 2: Get quantization config
    if quantization == "fp8":
        quant_cfg = mtq.FP8_DEFAULT_CFG.copy()
    elif quantization == "int4_awq":
        quant_cfg = mtq.INT4_AWQ_CFG.copy()
    # ... etc

    # Step 3: Run calibration (this takes time!)
    # Model sees 512 samples, learns min/max ranges
    model = mtq.quantize(model, quant_cfg, forward_loop=calib_loop)

    return model
```

### The Calibration Loop Explained

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CALIBRATION PROCESS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SAMPLE 1: "The quick brown fox..."                                         │
│  ════════════════════════════════════                                       │
│            │                                                                │
│            ▼                                                                │
│       ┌────────────────────┐                                               │
│       │ Forward pass       │                                               │
│       │ (no backward!)     │                                               │
│       └────────────────────┘                                               │
│            │                                                                │
│            ▼                                                                │
│       ┌────────────────────────────────────────────────────────┐           │
│       │ Record: Layer 0 attention weights range: [-0.5, 0.8]   │           │
│       │ Record: Layer 0 MLP weights range: [-1.2, 1.5]         │           │
│       │ ...                                                     │           │
│       └────────────────────────────────────────────────────────┘           │
│                                                                             │
│  SAMPLE 2: "Machine learning is..."                                         │
│  SAMPLE 3: "The president announced..."                                     │
│  ... (512 samples total)                                                    │
│                                                                             │
│  AFTER ALL SAMPLES:                                                         │
│  ═══════════════════                                                        │
│       ┌────────────────────────────────────────────────────────┐           │
│       │ Layer 0 attention: seen range [-0.8, 1.2]              │           │
│       │ → Choose scale = 1.2 / 127 for INT8 quantization       │           │
│       │ → Or scale = 1.2 / 448 for FP8                         │           │
│       │                                                         │           │
│       │ Layer 0 MLP: seen range [-2.5, 3.1]                    │           │
│       │ → Choose scale = 3.1 / 127 for INT8                    │           │
│       └────────────────────────────────────────────────────────┘           │
│                                                                             │
│  WHY THIS MATTERS:                                                          │
│  • Too small scale → values overflow → garbage output                       │
│  • Too large scale → wasted precision → quality loss                        │
│  • Just right → minimal quality loss, maximum efficiency                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Quantization Schemes Compared

```python
# Available quantization methods in TensorRT-Edge-LLM

schemes = {
    "fp8": {
        "config": mtq.FP8_DEFAULT_CFG,
        "memory": "50% of FP16",
        "quality": "~99% of FP16",
        "speed": "1.5x faster than FP16",
        "best_for": "Default choice, best quality/size tradeoff"
    },
    "int4_awq": {
        "config": mtq.INT4_AWQ_CFG,
        "memory": "25% of FP16",
        "quality": "~97-98% of FP16",
        "speed": "2x faster than FP16",
        "best_for": "Memory-constrained devices"
    },
    "nvfp4": {
        "config": mtq.NVFP4_DEFAULT_CFG,
        "memory": "25% of FP16",
        "quality": "~97-98% of FP16",
        "speed": "2x faster than FP16",
        "best_for": "Newer GPUs with FP4 support"
    },
    "int8_sq": {
        "config": mtq.INT8_SMOOTHQUANT_CFG,
        "memory": "50% of FP16",
        "quality": "~98% of FP16",
        "speed": "1.5x faster than FP16",
        "best_for": "When FP8 isn't available"
    }
}
```

### Hands-on Exercise: Compare Quantization Quality

```python
# Exercise: Create a script to compare quantization schemes
# File: exercises/compare_quantization.py

"""
Task: Quantize the same model with different schemes and compare:
1. Model size on disk
2. Inference quality (perplexity or response quality)
3. Memory usage during inference

Steps:
1. Quantize with FP8, INT4_AWQ
2. Export both to ONNX
3. Compare file sizes
4. Run same prompts through both
5. Subjectively rate responses
"""

from tensorrt_edgellm import quantize_and_save_llm

model_dir = "Qwen/Qwen2.5-0.5B-Instruct"

# Scheme 1: FP8
quantize_and_save_llm(
    model_dir=model_dir,
    output_dir="./quantized_fp8",
    quantization="fp8"
)

# Scheme 2: INT4 AWQ
quantize_and_save_llm(
    model_dir=model_dir,
    output_dir="./quantized_int4",
    quantization="int4_awq"
)

# Compare sizes:
# ls -lh ./quantized_fp8/
# ls -lh ./quantized_int4/

# Q: What's the size difference?
# Q: Run the same prompt through both - can you tell the difference?
```

---

## 2. ONNX Export Internals

### What ONNX Export Does

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PYTORCH → ONNX CONVERSION                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PYTORCH: Dynamic, Python-based                                             │
│  ════════════════════════════════                                           │
│                                                                             │
│    model = LlamaForCausalLM(...)                                           │
│    output = model(input_ids)  # Python executes, traced at runtime         │
│                                                                             │
│    ┌─────────────────────────────────────────────────────────────────┐     │
│    │ Forward pass is PYTHON CODE                                      │     │
│    │ - Conditionals can change per input                             │     │
│    │ - Shapes can be dynamic                                         │     │
│    │ - Easy to debug, slow to execute                                │     │
│    └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│                              │                                              │
│                              │ torch.onnx.export()                          │
│                              ▼                                              │
│                                                                             │
│  ONNX: Static graph                                                         │
│  ═══════════════════                                                        │
│                                                                             │
│    ┌─────────────────────────────────────────────────────────────────┐     │
│    │ model.onnx                                                       │     │
│    │                                                                  │     │
│    │ Node 0: Gather (embedding lookup)                               │     │
│    │   └─► Node 1: MatMul (Q projection)                             │     │
│    │         └─► Node 2: MatMul (K projection)                       │     │
│    │               └─► Node 3: MatMul (V projection)                 │     │
│    │                     └─► Node 4: Attention                       │     │
│    │                           └─► Node 5: MatMul (output proj)      │     │
│    │                                 └─► ...                         │     │
│    │                                                                  │     │
│    │ This is a FIXED GRAPH - no Python at runtime                    │     │
│    │ TensorRT can analyze and optimize this                          │     │
│    └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Export Process

```python
# File: tensorrt_edgellm/onnx_export/llm_export.py

def export_llm_model(model_dir, output_dir, device="cuda", ...):
    """
    Export flow:
    1. Load quantized model
    2. Wrap with EdgeLLM layers (optimized attention)
    3. Trace with example inputs
    4. Apply graph surgery
    5. Save ONNX + config files
    """

    # Step 1: Load quantized model
    model, tokenizer, processor = load_hf_model(model_dir, dtype, device)

    # Step 2: Wrap with EdgeLLM optimizations
    edge_model = EdgeLLMModelForCausalLM(
        model,
        is_eagle_base=is_eagle_base,
        use_prompt_tuning=is_vlm
    )

    # Step 3: Create example inputs for tracing
    example_inputs = create_example_inputs(
        batch_size=1,
        seq_len=128,
        hidden_size=model.config.hidden_size
    )

    # Step 4: Export to ONNX
    torch.onnx.export(
        edge_model,
        example_inputs,
        output_path,
        input_names=["input_ids", "position_ids", ...],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            ...
        }
    )

    # Step 5: Apply graph surgery (restructure for TensorRT)
    apply_onnx_surgery(output_path)

    # Step 6: Save config files
    save_model_config(model.config, output_dir)
```

### Understanding Dynamic Axes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DYNAMIC AXES EXPLAINED                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PROBLEM: ONNX graphs have fixed shapes by default                          │
│                                                                             │
│    Export with batch=1, seq=128                                             │
│    → Can ONLY run with batch=1, seq=128                                     │
│    → Need different model for batch=4!                                      │
│                                                                             │
│  SOLUTION: Dynamic axes                                                     │
│                                                                             │
│    dynamic_axes = {                                                         │
│        "input_ids": {0: "batch", 1: "seq_len"},                            │
│        "attention_mask": {0: "batch", 1: "seq_len"},                       │
│        "logits": {0: "batch"}                                              │
│    }                                                                        │
│                                                                             │
│    → Model works with ANY batch size, ANY sequence length                   │
│    → TensorRT optimizes for specific ranges at build time                   │
│                                                                             │
│  EXAMPLE:                                                                   │
│                                                                             │
│    input_ids shape: [batch, seq_len]                                        │
│                       │       │                                             │
│                       │       └─ Dynamic: can be 1, 128, 512, 2048...      │
│                       └─ Dynamic: can be 1, 2, 4...                        │
│                                                                             │
│    At TensorRT build time:                                                  │
│    --maxBatchSize 4          → batch ∈ [1, 4]                              │
│    --maxInputLen 2048        → seq_len ∈ [1, 2048]                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Graph Surgery

### What Graph Surgery Does

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GRAPH SURGERY: BEFORE/AFTER                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  BEFORE SURGERY (naive ONNX export):                                        │
│  ════════════════════════════════════                                       │
│                                                                             │
│    input_ids                                                                │
│        │                                                                    │
│        ▼                                                                    │
│    [Gather] ─────────────► embedding                                        │
│        │                                                                    │
│        ▼                                                                    │
│    [MatMul] ─────────────► Q                                               │
│        │                                                                    │
│        ▼                                                                    │
│    [MatMul] ─────────────► K                                               │
│        │                                                                    │
│        ▼                                                                    │
│    [MatMul] ─────────────► V                                               │
│        │                                                                    │
│        ▼                                                                    │
│    [Reshape]                                                                │
│        │                                                                    │
│        ▼                                                                    │
│    [Transpose]                                                              │
│        │                                                                    │
│        ▼                                                                    │
│    [MatMul] ─────────────► QK^T                                            │
│        │                                                                    │
│        ▼                                                                    │
│    [Softmax]                                                                │
│        │                                                                    │
│        ▼                                                                    │
│    [MatMul] ─────────────► attention output                                │
│        │                                                                    │
│        ▼                                                                    │
│    [Transpose]                                                              │
│        │                                                                    │
│        ▼                                                                    │
│    [Reshape]                                                                │
│        │                                                                    │
│        ▼                                                                    │
│    ... (many more nodes)                                                    │
│                                                                             │
│  AFTER SURGERY (optimized for TensorRT):                                    │
│  ════════════════════════════════════════                                   │
│                                                                             │
│    input_ids                                                                │
│        │                                                                    │
│        ▼                                                                    │
│    [Gather] ─────────────► embedding                                        │
│        │                                                                    │
│        ▼                                                                    │
│    ┌─────────────────────────────────────────┐                             │
│    │         ATTENTION PLUGIN                 │                             │
│    │  (replaces ~20 nodes with 1 custom op)  │                             │
│    │                                          │                             │
│    │  • Fused Q, K, V projection             │                             │
│    │  • Fused RoPE positional encoding       │                             │
│    │  • Optimized attention computation      │                             │
│    │  • KV cache read/write                  │                             │
│    └─────────────────────────────────────────┘                             │
│        │                                                                    │
│        ▼                                                                    │
│    output                                                                   │
│                                                                             │
│  BENEFIT: Fewer kernel launches, less memory traffic                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Surgery Operations

```python
# File: tensorrt_edgellm/onnx_export/onnx_utils.py

def apply_onnx_surgery(model_path):
    """
    Surgery operations:
    1. Replace attention pattern with custom plugin
    2. Fuse layer normalization
    3. Remove unnecessary ops (cast, reshape identity)
    4. Optimize constant folding
    """

    # Load ONNX model
    model = onnx.load(model_path)

    # Surgery 1: Replace attention with plugin
    model = replace_attention_with_plugin(model)

    # Surgery 2: Fuse layer norm
    model = fuse_layer_norm(model)

    # Surgery 3: Remove identity ops
    model = remove_identity_ops(model)

    # Surgery 4: Constant folding
    model = fold_constants(model)

    # Save optimized model
    onnx.save(model, model_path)
```

---

## 4. Adding New Model Support

### The Model Detection System

```python
# File: tensorrt_edgellm/llm_models/model_utils.py

def _check_model_type(model_dir: str, identifier: str) -> bool:
    """
    How TensorRT-Edge-LLM knows which model you're using:

    1. Load config.json from model directory
    2. Check 'model_type' field
    3. Check 'architectures' field

    Example config.json:
    {
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": 4096,
        ...
    }

    identifier="qwen" → matches "qwen2" → returns True
    """
    cfg = AutoConfig.from_pretrained(model_dir)

    # Check model_type
    model_type = str(getattr(cfg, "model_type", "")).lower()
    if identifier in model_type:
        return True

    # Check architectures
    archs = getattr(cfg, "architectures", []) or []
    return any(identifier in str(a).lower() for a in archs)
```

### Adding a New Model: Checklist

```markdown
## New Model Support Checklist

### Step 1: Does it work automatically?
Try the standard pipeline first:
```bash
tensorrt-edgellm-quantize-llm \
    --model_dir YourNewModel \
    --output_dir ./test \
    --quantization fp8
```
If it works → you're done! Most HuggingFace models work automatically.

### Step 2: Check for errors
Common issues:
- [ ] Missing chat template → Add to `tensorrt_edgellm/chat_templates/`
- [ ] Special attention pattern → May need layer modifications
- [ ] Custom activation function → Add to `llm_models/layers/`

### Step 3: Add chat template (if needed)
File: `tensorrt_edgellm/chat_templates/your_model.json`
```json
{
    "model_type": "your_model",
    "bos_token": "<|begin|>",
    "eos_token": "<|end|>",
    "template": "<|user|>{user_message}<|assistant|>"
}
```

### Step 4: Add tests
File: `tests/test_lists/l0_pipeline_a30.yml`
```yaml
- tests/defs/test_model_export.py::test_model_export[YourModel-fp16-mxsl4096-mxbs1-mxil2048]
```

### Step 5: Update documentation
File: `docs/source/developer_guide/02_Supported_Models.md`
Add your model to the supported list.
```

### Hands-on Exercise: Trace Model Loading

```python
# Exercise: Understand how models are loaded
# Add print statements to trace the flow

# File: tensorrt_edgellm/llm_models/model_utils.py

def load_hf_model(model_dir, dtype, device):
    print(f"Step 1: Loading config from {model_dir}")
    config = AutoConfig.from_pretrained(model_dir)
    print(f"  model_type: {config.model_type}")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_layers: {config.num_hidden_layers}")

    print(f"Step 2: Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print(f"  vocab_size: {tokenizer.vocab_size}")

    print(f"Step 3: Loading model weights")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map=device
    )
    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, tokenizer, None

# Run with:
# python -c "from tensorrt_edgellm.llm_models.model_utils import load_hf_model; load_hf_model('Qwen/Qwen2.5-0.5B-Instruct', 'float16', 'cpu')"
```

---

## Feynman Self-Test

- [ ] **What does quantization do in one sentence?**
  > Reduces weight precision (bits per number) to save memory while maintaining model quality.

- [ ] **Why do we need calibration?**
  > To find the range of values each layer produces, so we can choose optimal scale factors for quantization.

- [ ] **What's the difference between PyTorch model and ONNX?**
  > PyTorch is dynamic Python code; ONNX is a static graph that any runtime can execute.

- [ ] **What is graph surgery?**
  > Restructuring the ONNX graph to replace many small ops with fewer, fused custom ops optimized for TensorRT.

- [ ] **How does the system know which model type you're using?**
  > It reads config.json and checks model_type and architectures fields.

## If You're Stuck

### "Quantization fails with OOM"
Reduce calibration samples:
```python
quantize_and_save_llm(..., num_calib_samples=128)  # Down from 512
```

### "ONNX export hangs"
The model might have unsupported ops. Check:
```python
# Try exporting without surgery first
torch.onnx.export(model, inputs, "test.onnx", opset_version=17)
# Then check test.onnx with Netron (visual ONNX viewer)
```

### "New model doesn't work"
Check what's different:
```python
config_old = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
config_new = AutoConfig.from_pretrained("YourNewModel")
# Compare: hidden_size, num_attention_heads, etc.
```

---

## What's Next?

You now understand:
- ✅ How quantization works and why we calibrate
- ✅ ONNX export process and dynamic axes
- ✅ Graph surgery optimizations
- ✅ How to add new model support

**Next**: [05 C++ Runtime Internals](05_cpp_runtime.md) - Deep dive into engine execution, the decode loop, and attention kernels.

---

*← [03 Memory Model](03_memory_model.md) | [05 C++ Runtime →](05_cpp_runtime.md)*
