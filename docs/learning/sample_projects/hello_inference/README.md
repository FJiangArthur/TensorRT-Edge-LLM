# Project: Hello Inference

**Difficulty**: Beginner
**Estimated Time**: 30 minutes
**Prerequisites**: Level 1 completed

## Goal

Run your first complete inference pipeline from scratch:
1. Quantize a small model
2. Export to ONNX
3. Build TensorRT engine
4. Run inference

## Steps

### Step 1: Setup Environment

```bash
# Create project directory
mkdir -p workspace/hello_inference
cd workspace/hello_inference

# Copy this project's files
cp -r /path/to/sample_projects/hello_inference/* .
```

### Step 2: Quantize Model

We'll use the smallest model for quick iteration:

```bash
tensorrt-edgellm-quantize-llm \
    --model_dir Qwen/Qwen2.5-0.5B-Instruct \
    --output_dir ./step1_quantized \
    --quantization fp8

# Expected output:
# - ./step1_quantized/config.json
# - ./step1_quantized/model.safetensors
```

**What happened?**
- Downloaded model from HuggingFace (if not cached)
- Converted FP16 weights to FP8 (8-bit float)
- Saved quantized model

### Step 3: Export to ONNX

```bash
tensorrt-edgellm-export-llm \
    --model_dir ./step1_quantized \
    --output_dir ./step2_onnx \
    --max_batch_size 1 \
    --max_input_length 256 \
    --max_output_length 512

# Expected output:
# - ./step2_onnx/decoder_*.onnx
# - ./step2_onnx/config.json
# - ./step2_onnx/tokenizer.json
```

**What happened?**
- Traced PyTorch model execution
- Applied graph surgery (optimizations for TensorRT)
- Saved ONNX files with dynamic axes

### Step 4: Build TensorRT Engine

```bash
./build/examples/llm/llm_build \
    --onnxDir ./step2_onnx \
    --engineDir ./step3_engine \
    --maxBatchSize 1 \
    --maxInputLen 256 \
    --maxKVCacheCapacity 512

# Expected output:
# - ./step3_engine/decoder.plan
# - ./step3_engine/config.json
```

**What happened?**
- TensorRT analyzed ONNX graph
- Fused layers, selected optimal kernels
- Compiled for your specific GPU

### Step 5: Run Inference

Create `input.json`:
```json
{
    "batch_size": 1,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_generate_length": 100,
    "requests": [
        {
            "messages": [
                {"role": "user", "content": "What is 2+2? Answer in one sentence."}
            ]
        }
    ]
}
```

Run:
```bash
./build/examples/llm/llm_inference \
    --engineDir ./step3_engine \
    --inputFile input.json \
    --outputFile output.json

# View result
cat output.json | python -m json.tool
```

## Exercises

### Exercise 1: Change the Prompt

Modify `input.json` with your own questions:
```json
{
    "messages": [
        {"role": "user", "content": "Write a haiku about programming."}
    ]
}
```

### Exercise 2: Adjust Sampling

Try different sampling parameters:
```json
{
    "temperature": 0.1,    // More deterministic
    "top_k": 10,           // Limit vocabulary
    "max_generate_length": 200
}
```

### Exercise 3: Add System Prompt

```json
{
    "requests": [
        {
            "system_prompt": "You are a helpful coding assistant.",
            "messages": [
                {"role": "user", "content": "How do I reverse a list in Python?"}
            ]
        }
    ]
}
```

### Exercise 4: Multi-turn Conversation

```json
{
    "requests": [
        {
            "messages": [
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Nice to meet you, Alice!"},
                {"role": "user", "content": "What's my name?"}
            ]
        }
    ]
}
```

## Understanding the Output

```json
{
    "responses": [
        {
            "output_text": "2+2 equals 4.",
            "tokens_generated": 5,
            "prefill_time_ms": 45.2,      // Time to process input
            "decode_time_ms": 120.5,      // Time to generate output
            "tokens_per_second": 41.5     // Generation speed
        }
    ]
}
```

## Troubleshooting

### "Command not found"
```bash
# Ensure Python package is installed
pip install -e /path/to/TensorRT-Edge-LLM

# Ensure C++ binaries are built
ls ./build/examples/llm/
```

### "CUDA out of memory"
```bash
# Use smaller settings
--maxKVCacheCapacity 256
--maxInputLen 128
```

### "Model not found"
```bash
# Check HuggingFace login
huggingface-cli login
```

## What You Learned

- ✅ The complete four-step pipeline
- ✅ How to configure inference parameters
- ✅ How to interpret output metrics

## Next Steps

- Try a larger model (Qwen2.5-3B)
- Try different quantization (int4_awq)
- Move to [multi_adapter](../multi_adapter/) project
