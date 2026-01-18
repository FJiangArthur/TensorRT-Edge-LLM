# Level 2a: Architecture Deep Dive

**Reading Time: 1.5 hours**

Imagine a factory assembly line where raw materials enter one end and finished products exit the other. Each station does one specific job, and they're arranged so materials never travel backwards. TensorRT-Edge-LLM is exactly this: a carefully designed pipeline where your prompt enters, flows through specialized processing stages, and emerges as generated text. Understanding this flow is the key to understanding (and eventually modifying) the system.

## The Core Insight First

1. **The system is a directed acyclic graph (DAG)** - Data flows forward only, never backwards
2. **Python handles model surgery, C++ handles execution** - Each language does what it's best at
3. **The decode loop is the heart** - Everything else exists to make this loop fast

## Key Numbers to Memorize

| Component | Latency/Size | Why It Matters |
|-----------|--------------|----------------|
| **Tokenization** | <1ms | Negligible, but errors here break everything |
| **Prefill** | 50-500ms | Scales linearly with input length |
| **Per-token decode** | 30-50ms | Your tokens-per-second = 1000/this |
| **Model weights** | 0.5-2 bytes/param | 7B model ≈ 7-14GB |
| **KV cache per token** | ~1MB per 7B model | 1000 tokens ≈ 1GB cache |

## Table of Contents

1. [The Complete Data Flow](#1-the-complete-data-flow) (20 min)
2. [Component Deep Dive](#2-component-deep-dive) (40 min)
3. [The Decode Loop Explained](#3-the-decode-loop-explained) (20 min)
4. [Configuration Impact](#4-configuration-impact) (10 min)

---

## 1. The Complete Data Flow

### End-to-End Journey of a Prompt

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPLETE DATA FLOW                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  USER PROMPT: "Explain gravity"                                             │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ TOKENIZER                                                            │   │
│  │ "Explain gravity" → [849, 1601, 9324, 15]                           │   │
│  │ Role: Convert text to numbers the model understands                 │   │
│  │ File: cpp/tokenizer/tokenizer.cpp                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼ [4 tokens]                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ CHAT TEMPLATE                                                        │   │
│  │ Add special tokens: <|user|> + tokens + <|assistant|>               │   │
│  │ Role: Format input according to model's training                    │   │
│  │ File: tensorrt_edgellm/chat_templates/                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼ [~10 tokens with template]                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ EMBEDDING LOOKUP                                                     │   │
│  │ token_ids → dense vectors [seq_len, hidden_size]                    │   │
│  │ Role: Convert discrete tokens to continuous representations         │   │
│  │ Kernel: cpp/kernels/embeddingKernels/                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼ [10, 4096] tensor                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ PREFILL PHASE                                                        │   │
│  │ Process ALL input tokens through ALL transformer layers             │   │
│  │ Role: Build initial context understanding                           │   │
│  │ File: cpp/runtime/llmEngineRunner.cpp:executePrefillStep()          │   │
│  │                                                                      │   │
│  │   For each of 32 layers:                                            │   │
│  │     1. Self-Attention (Q, K, V computation)                         │   │
│  │     2. Feed-Forward Network (MLP)                                   │   │
│  │     3. Store K, V in cache                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ├──────────────────────────────────────────────┐                   │
│         ▼                                              ▼                   │
│  ┌──────────────────┐                    ┌─────────────────────────────┐   │
│  │ OUTPUT LOGITS    │                    │ KV CACHE (filled)           │   │
│  │ [1, vocab_size]  │                    │ [layers, batch, 2, heads,   │   │
│  │ Probabilities    │                    │  seq_len, head_dim]         │   │
│  │ for next token   │                    │                             │   │
│  └──────────────────┘                    └─────────────────────────────┘   │
│         │                                              │                   │
│         ▼                                              │                   │
│  ┌─────────────────────────────────────────────────────┴───────────────┐   │
│  │ DECODE LOOP (repeat until done)                                      │   │
│  │                                                                      │   │
│  │   ┌───────────────────────────────────────────────────────────────┐ │   │
│  │   │ 1. SAMPLE: Pick next token from logits                        │ │   │
│  │   │    top_k=40, top_p=0.9, temperature=0.7                       │ │   │
│  │   │    File: cpp/sampler/samplingKernels.cu                       │ │   │
│  │   └───────────────────────────────────────────────────────────────┘ │   │
│  │                         │                                           │   │
│  │                         ▼ [1 token]                                 │   │
│  │   ┌───────────────────────────────────────────────────────────────┐ │   │
│  │   │ 2. EMBED: Look up embedding for new token                     │ │   │
│  │   └───────────────────────────────────────────────────────────────┘ │   │
│  │                         │                                           │   │
│  │                         ▼ [1, hidden_size]                          │   │
│  │   ┌───────────────────────────────────────────────────────────────┐ │   │
│  │   │ 3. ATTENTION: Compute attention with cached K, V              │ │   │
│  │   │    Q: new token                                               │ │   │
│  │   │    K, V: from cache (all previous tokens)                     │ │   │
│  │   │    File: cpp/kernels/decodeAttentionKernels/                  │ │   │
│  │   └───────────────────────────────────────────────────────────────┘ │   │
│  │                         │                                           │   │
│  │                         ▼                                           │   │
│  │   ┌───────────────────────────────────────────────────────────────┐ │   │
│  │   │ 4. UPDATE: Add new K, V to cache                              │ │   │
│  │   │    Cache grows by 1 token per decode step                     │ │   │
│  │   └───────────────────────────────────────────────────────────────┘ │   │
│  │                         │                                           │   │
│  │                         ▼                                           │   │
│  │   ┌───────────────────────────────────────────────────────────────┐ │   │
│  │   │ 5. OUTPUT: New logits → back to step 1                        │ │   │
│  │   │    Until: max_length OR end_token                             │ │   │
│  │   └───────────────────────────────────────────────────────────────┘ │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼ [sequence of token_ids]                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ DETOKENIZER                                                          │   │
│  │ [849, 1601, ...] → "Gravity is a force that..."                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  OUTPUT: "Gravity is a force that attracts objects with mass..."           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Time Breakdown (100-token generation)

```
┌───────────────────────────────────────────────────────────────┐
│                    TIME BREAKDOWN                             │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  Tokenize:        ████  1ms                                   │
│                                                               │
│  Prefill:         ████████████████████  200ms                 │
│                   (processes 50 input tokens)                 │
│                                                               │
│  Decode x 100:    ████████████████████████████████████████    │
│                   ████████████████████████████████████████    │
│                   ████████████████████  4000ms                │
│                   (40ms per token × 100 tokens)               │
│                                                               │
│  Detokenize:      ██  0.5ms                                   │
│                                                               │
│  TOTAL:           ~4.2 seconds                                │
│                                                               │
│  Where optimization matters:                                  │
│  • Decode is 95% of time → optimize decode loop              │
│  • Prefill is 5% but blockin → optimize first token latency │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

## 2. Component Deep Dive

### 2.1 The Tokenizer

**What it does**: Converts text to numbers and back.

```python
# Conceptual example (not actual API)
tokenizer = load_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")

# Encoding: text → tokens
tokens = tokenizer.encode("Hello world")  # [15496, 1917]

# Decoding: tokens → text
text = tokenizer.decode([15496, 1917])    # "Hello world"
```

**Key insight**: Tokenizers are trained alongside models. Using the wrong tokenizer = garbage output.

**File locations**:
- Python: HuggingFace `transformers` library
- C++: `cpp/tokenizer/tokenizer.cpp`

### 2.2 The Embedding Table

**What it does**: Looks up dense vectors for each token.

```
Token ID: 15496
    │
    ▼
┌───────────────────────────────────────┐
│     EMBEDDING TABLE                   │
│     [vocab_size, hidden_size]         │
│     [151936, 4096]                    │
│                                       │
│     Row 0:     [0.12, -0.34, ...]    │
│     Row 1:     [0.56, 0.78, ...]     │
│     ...                               │
│     Row 15496: [0.23, -0.11, ...]  ◄──┤ This row!
│     ...                               │
│     Row 151935: [0.89, 0.12, ...]    │
└───────────────────────────────────────┘
    │
    ▼
Output: [0.23, -0.11, 0.45, ...] (4096 values)
```

**Key numbers**:
- Vocab size: ~32K-150K tokens
- Hidden size: 2048-8192 dimensions
- Memory: vocab_size × hidden_size × 2 bytes (FP16)

### 2.3 The Transformer Layer

**What it does**: Processes embeddings through attention and feed-forward networks.

```
                    Input: [batch, seq_len, hidden_size]
                                    │
                                    ▼
              ┌─────────────────────────────────────────┐
              │            SELF-ATTENTION               │
              │                                         │
              │   Q = input × W_q    (Query)           │
              │   K = input × W_k    (Key)             │
              │   V = input × W_v    (Value)           │
              │                                         │
              │   Attention = softmax(Q × K^T / √d) × V│
              │                                         │
              └─────────────────────────────────────────┘
                                    │
                                    ▼
              ┌─────────────────────────────────────────┐
              │            FEED-FORWARD (MLP)          │
              │                                         │
              │   hidden = input × W_gate × W_up       │
              │   output = SiLU(hidden) × W_down       │
              │                                         │
              └─────────────────────────────────────────┘
                                    │
                                    ▼
                    Output: [batch, seq_len, hidden_size]
```

**Repeated 32 times** (for a typical 7B model)

### 2.4 The KV Cache

**What it does**: Stores computed K and V values to avoid recomputation.

```
WITHOUT CACHE (naive):
  Token 1: Compute K1, V1
  Token 2: Compute K1, V1, K2, V2  ← Recomputing K1, V1!
  Token 3: Compute K1, V1, K2, V2, K3, V3  ← Recomputing again!

WITH CACHE (efficient):
  Token 1: Compute K1, V1 → Store in cache
  Token 2: Load K1, V1 from cache, compute K2, V2 → Store
  Token 3: Load K1, V1, K2, V2 from cache, compute K3, V3 → Store

Speedup: O(n²) → O(n)
```

**Memory layout**:
```
Shape: [num_layers, batch_size, 2, num_kv_heads, max_seq_len, head_dim]

Example for 7B model:
[32 layers] × [1 batch] × [2 for K,V] × [8 heads] × [4096 seq] × [128 dim]
= 32 × 1 × 2 × 8 × 4096 × 128 = 268 million values
= 536 MB in FP16
```

### 2.5 The Sampler

**What it does**: Converts logits to the next token.

```
Logits: [vocab_size] raw scores
         │
         ▼
    ┌────────────────────────────────────────┐
    │  1. Apply temperature (scale)          │
    │     logits = logits / temperature      │
    │     Higher temp = more random          │
    └────────────────────────────────────────┘
         │
         ▼
    ┌────────────────────────────────────────┐
    │  2. Apply top_k (filter)               │
    │     Keep only top 40 highest logits    │
    │     Set rest to -infinity              │
    └────────────────────────────────────────┘
         │
         ▼
    ┌────────────────────────────────────────┐
    │  3. Apply softmax (normalize)          │
    │     Convert to probabilities           │
    │     Sum = 1.0                          │
    └────────────────────────────────────────┘
         │
         ▼
    ┌────────────────────────────────────────┐
    │  4. Apply top_p (nucleus sampling)     │
    │     Keep tokens until cumsum > 0.9     │
    └────────────────────────────────────────┘
         │
         ▼
    ┌────────────────────────────────────────┐
    │  5. Sample (random selection)          │
    │     Pick one token based on probs      │
    └────────────────────────────────────────┘
         │
         ▼
    Output: single token_id
```

---

## 3. The Decode Loop Explained

### Why Decode is Sequential

**The fundamental constraint**: Each token depends on ALL previous tokens.

```
Generating: "The cat sat on the"

Step 1: Input="The" → Output=probabilities → Sample="cat"
Step 2: Input="The cat" → Output=probabilities → Sample="sat"
Step 3: Input="The cat sat" → Output=probabilities → Sample="on"
Step 4: Input="The cat sat on" → Output=probabilities → Sample="the"
...

Each step MUST see all previous tokens.
This is why LLM generation is inherently sequential.
```

### The KV Cache Optimization

**Without cache**: Recompute attention for ALL tokens every step.
**With cache**: Compute attention only for NEW token, reuse cached K, V.

```
Step 1:
  Q = embed("The")           # [1, hidden]
  K = embed("The")           # [1, hidden] → cache
  V = embed("The")           # [1, hidden] → cache
  Attention = Q × K^T × V

Step 2:
  Q = embed("cat")           # [1, hidden] - only new token!
  K = cache + embed("cat")   # [2, hidden] - append to cache
  V = cache + embed("cat")   # [2, hidden] - append to cache
  Attention = Q × K^T × V    # Q is [1], K,V are [2]

Step 3:
  Q = embed("sat")           # [1, hidden]
  K = cache + embed("sat")   # [3, hidden]
  V = cache + embed("sat")   # [3, hidden]
  Attention = Q × K^T × V    # Q is [1], K,V are [3]

...
```

### Decode Attention vs Context Attention

| Aspect | Context (Prefill) | Decode |
|--------|-------------------|--------|
| **Q shape** | [batch, seq_len, heads, dim] | [batch, 1, heads, dim] |
| **Purpose** | Process all input | Generate one token |
| **Kernel** | FMHA (Flash Attention) | XQA (Extended Query) |
| **GPU utilization** | High (parallel) | Lower (sequential) |
| **File** | `contextAttentionKernels/` | `decodeAttentionKernels/` |

---

## 4. Configuration Impact

### Build-Time vs Run-Time Configuration

| Parameter | Set When | Impact |
|-----------|----------|--------|
| `maxBatchSize` | Build | Memory allocation |
| `maxInputLen` | Build | Maximum prompt length |
| `maxKVCacheCapacity` | Build | Maximum total sequence |
| `temperature` | Run | Output randomness |
| `top_k`, `top_p` | Run | Sampling strategy |
| `max_generate_length` | Run | Output length limit |

### The Memory Equation

```
Total GPU Memory = Weights + KV Cache + Activations + Workspace

For 7B model with 4K context:
  Weights:     ~14GB (FP16) or ~7GB (FP8)
  KV Cache:    ~0.5GB per 1K tokens
  Activations: ~1-2GB (during prefill)
  Workspace:   ~0.5GB

  Minimum: ~16GB for FP16, ~10GB for FP8
```

### Hands-on Exercise: Trace a Request

Open the example inference code and trace one request:

```bash
# File: examples/llm/llm_inference.cpp
# Find these functions and note what they do:

1. parseInputRequest()    - Line ~XXX: How is JSON parsed?
2. handleRequest()        - Line ~XXX: Main entry point
3. executePrefillStep()   - Line ~XXX: Prefill phase
4. executeVanillaDecodingStep() - Line ~XXX: Decode loop
```

Create a diagram showing:
- Input: What enters each function?
- Output: What leaves each function?
- Side effects: What state changes?

---

## Feynman Self-Test

- [ ] **Can I draw the data flow from prompt to output?**
  > Tokenize → Embed → Prefill (all layers) → Decode loop (sample → embed → attention → update cache) → Detokenize

- [ ] **Why is KV cache important?**
  > Avoids recomputing K, V for all previous tokens; reduces complexity from O(n²) to O(n)

- [ ] **Where does most time go in generation?**
  > Decode loop: 95%+ for longer outputs

- [ ] **What's the difference between prefill and decode attention?**
  > Prefill: Q is [seq_len], processes all at once; Decode: Q is [1], processes one token

- [ ] **What determines maximum context length?**
  > maxKVCacheCapacity setting at build time (determines pre-allocated cache size)

## If You're Stuck

### "I don't understand attention"
- Focus on the KV cache first - it's the key optimization
- Think of K as "what this token contains", V as "what to return if matched"
- Q asks "what am I looking for?"

### "The data shapes are confusing"
```
[batch, seq_len, heads, head_dim] is the main pattern
  batch=1 for single request
  seq_len varies (input length for prefill, 1 for decode)
  heads=32 typically (number of attention heads)
  head_dim=128 typically (hidden_size / heads)
```

### "I can't find the code"
Use grep:
```bash
grep -r "executePrefillStep" cpp/
grep -r "KVCache" cpp/runtime/
```

---

## What's Next?

You now understand:
- ✅ Complete data flow from prompt to output
- ✅ Role of each component
- ✅ Why the decode loop is the bottleneck

**Next**: [03 Memory Model](03_memory_model.md) - Deep dive into GPU memory management, KV cache internals, and memory optimization.

---

*← [01 Getting Started](01_getting_started.md) | [03 Memory Model →](03_memory_model.md)*
