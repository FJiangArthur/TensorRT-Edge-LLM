# Project: Multi-Adapter LoRA Switching

**Difficulty**: Intermediate
**Estimated Time**: 1 hour
**Prerequisites**: Levels 1-4 completed

## Goal

Build a system that can switch between multiple LoRA adapters at runtime:
1. Export model with LoRA support
2. Prepare multiple adapters
3. Switch adapters based on input domain
4. Compare outputs between adapters

## Concept

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTI-ADAPTER SYSTEM                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User Input: "What are the side effects of aspirin?"           │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────┐                                               │
│  │  Classifier │  → Domain: "medical"                          │
│  └─────────────┘                                               │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Base Model (7B)                            │   │
│  │  + LoRA Adapter: medical_qa.safetensors                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│  "Aspirin can cause stomach irritation, bleeding..."           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Steps

### Step 1: Export with LoRA Support

```bash
# Quantize base model
tensorrt-edgellm-quantize-llm \
    --model_dir Qwen/Qwen2.5-3B-Instruct \
    --output_dir ./base_quantized \
    --quantization fp8

# Export with LoRA support (max_lora_rank determines adapter capacity)
tensorrt-edgellm-export-llm \
    --model_dir ./base_quantized \
    --output_dir ./base_onnx \
    --max_lora_rank 64 \
    --max_batch_size 1

# Build engine
./build/examples/llm/llm_build \
    --onnxDir ./base_onnx \
    --engineDir ./engine \
    --maxBatchSize 1 \
    --maxKVCacheCapacity 2048
```

### Step 2: Prepare LoRA Adapters

For this exercise, we'll create mock adapters. In production, you'd train real adapters.

Create `prepare_mock_adapters.py`:
```python
#!/usr/bin/env python3
"""
Create mock LoRA adapters for testing.
In production, these would come from actual fine-tuning.
"""

import torch
from safetensors.torch import save_file
import os

def create_mock_adapter(name: str, rank: int = 64, hidden_size: int = 2048):
    """Create a mock LoRA adapter with random weights."""

    # Typical layers that get LoRA adapters
    layers = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]

    tensors = {}
    num_layers = 26  # Qwen2.5-3B has 26 layers

    for layer_idx in range(num_layers):
        for proj in layers:
            # LoRA A: [in_features, rank]
            # LoRA B: [rank, out_features]

            if proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                in_features = hidden_size
                out_features = hidden_size
            else:  # MLP layers have different sizes
                in_features = hidden_size
                out_features = hidden_size * 4 if proj != "down_proj" else hidden_size

            # Random initialization (real training would learn these)
            lora_a = torch.randn(in_features, rank, dtype=torch.float16) * 0.01
            lora_b = torch.randn(rank, out_features, dtype=torch.float16) * 0.01

            key_prefix = f"model.layers.{layer_idx}.self_attn.{proj}" if "proj" in proj[:1] else \
                        f"model.layers.{layer_idx}.mlp.{proj}"

            tensors[f"{key_prefix}.lora_A.weight"] = lora_a
            tensors[f"{key_prefix}.lora_B.weight"] = lora_b

    # Save adapter
    os.makedirs(f"adapters/{name}", exist_ok=True)
    save_file(tensors, f"adapters/{name}/adapter.safetensors")
    print(f"Created adapter: adapters/{name}/adapter.safetensors")
    print(f"  Layers: {num_layers}, Rank: {rank}")
    print(f"  Size: {sum(t.numel() * 2 for t in tensors.values()) / 1e6:.1f} MB")

if __name__ == "__main__":
    # Create mock adapters for different domains
    create_mock_adapter("medical", rank=64)
    create_mock_adapter("legal", rank=64)
    create_mock_adapter("technical", rank=64)
    print("\nMock adapters created. In production, train real adapters!")
```

Run:
```bash
python prepare_mock_adapters.py
```

### Step 3: Run with Adapter Switching

Create test inputs for different domains:

`medical_input.json`:
```json
{
    "batch_size": 1,
    "temperature": 0.7,
    "max_generate_length": 200,
    "requests": [
        {
            "system_prompt": "You are a medical information assistant.",
            "messages": [
                {"role": "user", "content": "What are common symptoms of the flu?"}
            ]
        }
    ]
}
```

`legal_input.json`:
```json
{
    "batch_size": 1,
    "temperature": 0.7,
    "max_generate_length": 200,
    "requests": [
        {
            "system_prompt": "You are a legal information assistant.",
            "messages": [
                {"role": "user", "content": "What is a non-disclosure agreement?"}
            ]
        }
    ]
}
```

Run with different adapters:
```bash
# Medical adapter
./build/examples/llm/llm_inference \
    --engineDir ./engine \
    --loraWeights "medical:adapters/medical/adapter.safetensors" \
    --inputFile medical_input.json \
    --outputFile medical_output.json

# Legal adapter
./build/examples/llm/llm_inference \
    --engineDir ./engine \
    --loraWeights "legal:adapters/legal/adapter.safetensors" \
    --inputFile legal_input.json \
    --outputFile legal_output.json

# Base model (no adapter)
./build/examples/llm/llm_inference \
    --engineDir ./engine \
    --inputFile medical_input.json \
    --outputFile base_output.json
```

### Step 4: Compare Outputs

```bash
echo "=== Medical Adapter ==="
cat medical_output.json | python -c "import sys,json; print(json.load(sys.stdin)['responses'][0]['output_text'])"

echo "\n=== Legal Adapter ==="
cat legal_output.json | python -c "import sys,json; print(json.load(sys.stdin)['responses'][0]['output_text'])"

echo "\n=== Base Model ==="
cat base_output.json | python -c "import sys,json; print(json.load(sys.stdin)['responses'][0]['output_text'])"
```

## Understanding the Implementation

### How LoRA Weights Are Bound

```cpp
// From llmEngineRunner.cpp
bool LLMEngineRunner::switchLoraWeights(
    const std::string& loraWeightsName,
    cudaStream_t stream)
{
    // Find adapter in loaded weights
    auto it = mLoraWeights.find(loraWeightsName);

    // Bind each tensor to TensorRT engine
    for (const auto& tensor : it->second) {
        setTensorAddress(tensor.getName(), tensor.rawPointer());
        setInputShape(tensor.getName(), tensor.getShape());
    }

    mActiveLoraWeightsName = loraWeightsName;
    return true;
}
```

### Why Switching is Fast

```
Traditional model switching:
  1. Unload model A (GPU → void)
  2. Load model B (disk → GPU)
  Time: ~30 seconds

LoRA switching:
  1. Update tensor pointers
  Time: <1 millisecond
```

## Exercises

### Exercise 1: Measure Switching Latency

Add timing around adapter switches:
```cpp
auto start = std::chrono::high_resolution_clock::now();
runner.switchLoraWeights("medical", stream);
cudaStreamSynchronize(stream);
auto end = std::chrono::high_resolution_clock::now();
std::cout << "Switch time: "
          << std::chrono::duration<double, std::milli>(end - start).count()
          << " ms" << std::endl;
```

### Exercise 2: Multiple Adapters in One Session

```bash
./build/examples/llm/llm_inference \
    --engineDir ./engine \
    --loraWeights "medical:adapters/medical/adapter.safetensors,legal:adapters/legal/adapter.safetensors,technical:adapters/technical/adapter.safetensors" \
    --inputFile combined_input.json
```

### Exercise 3: Adapter-Specific System Prompts

Create a routing system:
```python
def route_to_adapter(user_input: str) -> tuple[str, str]:
    """Return (adapter_name, system_prompt) based on input."""

    medical_keywords = ["symptom", "drug", "doctor", "pain", "health"]
    legal_keywords = ["contract", "law", "court", "rights", "agreement"]

    input_lower = user_input.lower()

    if any(kw in input_lower for kw in medical_keywords):
        return "medical", "You are a medical information assistant."
    elif any(kw in input_lower for kw in legal_keywords):
        return "legal", "You are a legal information assistant."
    else:
        return "", "You are a helpful assistant."  # Base model
```

## What You Learned

- ✅ How to export models with LoRA support
- ✅ How adapter switching works at runtime
- ✅ How to manage multiple adapters
- ✅ The performance characteristics of switching

## Next Steps

- Train real LoRA adapters using PEFT
- Build a web service with dynamic adapter routing
- Move to [benchmark_suite](../benchmark_suite/) project
