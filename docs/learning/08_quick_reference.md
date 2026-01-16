# Level 5a: Quick Reference Cards

**Purpose: Rapid lookup during development | Print-friendly**

This is your cheatsheet for day-to-day work with TensorRT-Edge-LLM. Bookmark this page.

---

## Command Cheatsheet

### Python Export Pipeline

```bash
# Step 1: Quantize model
tensorrt-edgellm-quantize-llm \
    --model_dir Qwen/Qwen2.5-7B-Instruct \
    --output_dir ./quantized \
    --quantization fp8              # Options: fp8, int4_awq, nvfp4, int8_sq

# Step 2: Export to ONNX
tensorrt-edgellm-export-llm \
    --model_dir ./quantized \
    --output_dir ./onnx \
    --max_batch_size 4 \
    --max_input_length 2048 \
    --max_output_length 4096

# Optional: Export with LoRA support
tensorrt-edgellm-export-llm \
    --model_dir ./quantized \
    --output_dir ./onnx \
    --max_lora_rank 64

# Optional: Export EAGLE draft model
tensorrt-edgellm-export-draft \
    --base_model_dir ./base_quantized \
    --draft_model_dir ./draft_quantized \
    --output_dir ./draft_onnx

# Optional: Export VLM vision encoder
tensorrt-edgellm-export-visual \
    --model_dir Qwen/Qwen2-VL-7B-Instruct \
    --output_dir ./vision_onnx

# Optional: Reduce vocabulary
tensorrt-edgellm-reduce-vocab \
    --model_dir Qwen/Qwen2.5-7B-Instruct \
    --output_dir ./reduced_vocab \
    --reduced_vocab_size 16384 \
    --method frequency             # Options: frequency, input_aware
```

### C++ Build & Run

```bash
# Build TensorRT engine
./build/examples/llm/llm_build \
    --onnxDir ./onnx \
    --engineDir ./engine \
    --maxBatchSize 4 \
    --maxInputLen 2048 \
    --maxKVCacheCapacity 4096

# Run standard inference
./build/examples/llm/llm_inference \
    --engineDir ./engine \
    --inputFile input.json \
    --outputFile output.json

# Run with LoRA
./build/examples/llm/llm_inference \
    --engineDir ./engine \
    --loraWeights "medical:medical.safetensors,legal:legal.safetensors" \
    --inputFile input.json

# Run EAGLE speculative decoding
./build/examples/llm/llm_inference_specdecode \
    --baseEngineDir ./base_engine \
    --draftEngineDir ./draft_engine \
    --inputFile input.json

# Run VLM inference
./build/examples/llm/llm_inference \
    --engineDir ./llm_engine \
    --multimodalEngineDir ./vision_engine \
    --inputFile input_with_images.json
```

### Build from Source

```bash
# Python package
pip install -e .

# C++ library
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run tests
pytest tests/ -v
./build/tests/test_runtime
```

---

## Input/Output Formats

### Standard Input (input.json)

```json
{
    "batch_size": 2,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "max_generate_length": 256,
    "requests": [
        {
            "system_prompt": "You are a helpful assistant.",
            "messages": [
                {"role": "user", "content": "What is 2+2?"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ]
        }
    ]
}
```

### VLM Input (with images)

```json
{
    "batch_size": 1,
    "max_generate_length": 512,
    "requests": [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Describe this image.",
                    "images": ["./photo.jpg"]
                }
            ]
        }
    ]
}
```

### Output Format

```json
{
    "responses": [
        {
            "output_text": "2+2 equals 4.",
            "tokens_generated": 5,
            "prefill_time_ms": 45.2,
            "decode_time_ms": 120.5,
            "tokens_per_second": 41.5
        }
    ]
}
```

---

## Configuration Reference

### Engine Build Options

| Option | Description | Default |
|--------|-------------|---------|
| `--maxBatchSize` | Maximum batch size | 1 |
| `--maxInputLen` | Maximum input tokens | 512 |
| `--maxOutputLen` | Maximum output tokens | 2048 |
| `--maxKVCacheCapacity` | Max KV cache entries | 4096 |
| `--useFP16` | Use FP16 precision | true |
| `--useCudaGraph` | Enable CUDA graphs | true |

### Quantization Options

| Scheme | Bits | When to Use |
|--------|------|-------------|
| `fp8` | 8 | Best quality, 2x compression |
| `int4_awq` | 4 | Good balance, 4x compression |
| `nvfp4` | 4 | Newer hardware, 4x compression |
| `int8_sq` | 8 | INT8 with SmoothQuant |

### Sampling Parameters

| Parameter | Range | Effect |
|-----------|-------|--------|
| `temperature` | 0.0-2.0 | Lower = deterministic, higher = creative |
| `top_p` | 0.0-1.0 | Nucleus sampling threshold |
| `top_k` | 1-100 | Top-K sampling limit |
| `repetition_penalty` | 1.0-2.0 | Penalize repeated tokens |

---

## Memory Calculation Formulas

### Model Weights

```
FP16: params × 2 bytes
FP8:  params × 1 byte
INT4: params × 0.5 bytes (+ scales overhead)

Example: 7B model
  FP16: 7B × 2 = 14 GB
  FP8:  7B × 1 = 7 GB
  INT4: 7B × 0.5 + scales ≈ 4 GB
```

### KV Cache

```
KV Cache = layers × batch × 2 × heads × seq_len × head_dim × dtype_size

Example: 7B model (32 layers, 8 KV heads, 128 head dim)
  FP16, batch=4, seq=4096:
  = 32 × 4 × 2 × 8 × 4096 × 128 × 2
  = 17 GB
```

### Total VRAM Budget

```
Total = Model Weights + KV Cache + Execution Context (~2GB) + CUDA Overhead (~1GB)

Example: 7B INT4 on Jetson Orin 32GB
  Weights: 4 GB
  KV Cache: 17 GB (with batch=4, seq=4096)
  Overhead: 3 GB
  Total: 24 GB ✓ Fits!
```

---

## Common Patterns

### Pattern 1: Minimal Setup

```bash
# Quickest path to running inference
tensorrt-edgellm-quantize-llm --model_dir Qwen/Qwen2.5-0.5B-Instruct \
    --output_dir ./q --quantization fp8
tensorrt-edgellm-export-llm --model_dir ./q --output_dir ./o
./llm_build --onnxDir ./o --engineDir ./e --maxBatchSize 1
./llm_inference --engineDir ./e --inputFile input.json
```

### Pattern 2: Production Setup (Memory Constrained)

```bash
# Optimized for Jetson Orin 16GB
tensorrt-edgellm-quantize-llm --model_dir Qwen/Qwen2.5-3B-Instruct \
    --output_dir ./quantized --quantization int4_awq

tensorrt-edgellm-export-llm --model_dir ./quantized --output_dir ./onnx \
    --max_batch_size 1 --max_input_length 1024 --max_output_length 2048

./llm_build --onnxDir ./onnx --engineDir ./engine \
    --maxBatchSize 1 --maxKVCacheCapacity 2048
```

### Pattern 3: Multi-Adapter Deployment

```bash
# Base model with LoRA support
tensorrt-edgellm-export-llm --model_dir ./quantized --output_dir ./onnx \
    --max_lora_rank 64

./llm_build --onnxDir ./onnx --engineDir ./engine

# Prepare adapters (outside TRT-Edge-LLM)
python prepare_lora.py --adapter medical --output medical.safetensors
python prepare_lora.py --adapter legal --output legal.safetensors

# Run with switching
./llm_inference --engineDir ./engine \
    --loraWeights "medical:medical.safetensors,legal:legal.safetensors"
```

### Pattern 4: High-Throughput with EAGLE

```bash
# Export base model for EAGLE
tensorrt-edgellm-export-llm --model_dir ./base_quantized --output_dir ./base_onnx \
    --eagle_base

# Export draft model
tensorrt-edgellm-export-draft --base_model_dir ./base_quantized \
    --draft_model_dir ./draft_quantized --output_dir ./draft_onnx

# Build both engines
./llm_build --onnxDir ./base_onnx --engineDir ./base_engine
./llm_build --onnxDir ./draft_onnx --engineDir ./draft_engine

# Run with speculative decoding (2-3x faster!)
./llm_inference_specdecode --baseEngineDir ./base_engine \
    --draftEngineDir ./draft_engine --inputFile input.json
```

---

## Key File Locations

### Python Package

```
tensorrt_edgellm/
├── scripts/
│   ├── quantize_llm.py      # Entry: tensorrt-edgellm-quantize-llm
│   └── export_llm.py        # Entry: tensorrt-edgellm-export-llm
├── llm_models/
│   ├── models/              # Model implementations
│   └── layers/              # Custom layers (INT4 GEMM, etc.)
├── quantization/            # Quantization algorithms
└── onnx_export/             # ONNX conversion
```

### C++ Runtime

```
cpp/
├── runtime/
│   ├── llmInferenceRuntime.h     # Main inference API
│   ├── llmEngineRunner.h         # TensorRT engine wrapper
│   └── linearKVCache.h           # KV cache implementation
├── kernels/
│   ├── contextAttentionKernels/  # Prefill attention
│   └── decodeAttentionKernels/   # Decode attention
└── common/
    └── tensor.h                   # Tensor class
```

### Generated Artifacts

```
my_model/
├── quantized/               # After quantize
│   ├── config.json
│   └── model.safetensors
├── onnx/                    # After export
│   ├── config.json
│   ├── decoder_*.onnx
│   └── tokenizer.json
└── engine/                  # After build
    ├── config.json
    └── decoder.plan
```

---

## Troubleshooting Quick Fixes

| Problem | Likely Cause | Fix |
|---------|--------------|-----|
| OOM during quantize | Model too large | Use `--device cpu` |
| OOM during build | KV cache too large | Reduce `--maxKVCacheCapacity` |
| OOM during inference | Batch too large | Reduce `--maxBatchSize` |
| Slow first token | No CUDA graphs | Ensure `useCudaGraph: true` |
| Wrong output | Tokenizer mismatch | Check chat template |
| Build fails | TensorRT version | Verify `trtexec --version` |

### Debug Commands

```bash
# Check GPU memory
nvidia-smi

# Check TensorRT version
trtexec --version

# Validate ONNX
python -c "import onnx; onnx.checker.check_model('model.onnx')"

# Profile engine
trtexec --loadEngine=decoder.plan --verbose

# Check Python package
pip show tensorrt-edgellm
```

---

## Performance Tuning Checklist

- [ ] **Quantization**: Using INT4/FP8 for memory-bound scenarios?
- [ ] **Batch size**: Maximized within memory budget?
- [ ] **KV cache**: Sized appropriately for context needs?
- [ ] **CUDA graphs**: Enabled for decode phase?
- [ ] **System prompt cache**: Using for repeated prompts?
- [ ] **EAGLE**: Considered for latency-sensitive apps?
- [ ] **Vocab reduction**: Applicable for domain-specific use?

---

## Key Numbers to Remember

| Metric | Value | Notes |
|--------|-------|-------|
| cudaMalloc latency | ~1ms | Pre-allocate everything |
| CUDA graph speedup | 10-30% | Decode phase only |
| INT4 compression | 4x | vs FP16 |
| EAGLE speedup | 2-3x | Generation throughput |
| LoRA switch time | <1ms | Just pointer updates |
| Vision tokens (Qwen) | 256-576 | Per image |
| KV cache per 7B | ~500MB/1K tokens | FP16, batch=1 |

---

*← [07 Advanced Features](07_advanced_features.md) | [09 Interview Prep →](09_interview_prep.md)*
