# Project: Benchmark Suite

**Difficulty**: Advanced
**Estimated Time**: 2 hours
**Prerequisites**: All levels completed

## Goal

Build a comprehensive benchmarking system to:
1. Measure and compare different configurations
2. Profile memory usage
3. Analyze latency breakdown
4. Generate performance reports

## Metrics We'll Measure

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **Prefill latency** | Time to process input | First token speed |
| **Decode latency** | Time per generated token | Generation speed |
| **Tokens/second** | Generation throughput | Overall performance |
| **Memory usage** | GPU VRAM consumed | Device compatibility |
| **First token latency** | Time to first output | User experience |

## Steps

### Step 1: Create Benchmark Script

Create `benchmark.py`:
```python
#!/usr/bin/env python3
"""
TensorRT-Edge-LLM Benchmarking Suite

Measures inference performance across different configurations.
"""

import subprocess
import json
import time
import os
from dataclasses import dataclass
from typing import List, Dict, Optional
import statistics


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    name: str
    engine_dir: str
    batch_size: int = 1
    input_length: int = 128
    output_length: int = 128
    num_runs: int = 5
    warmup_runs: int = 2


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    config_name: str
    prefill_latency_ms: float
    decode_latency_ms: float
    tokens_per_second: float
    first_token_latency_ms: float
    total_tokens: int
    memory_mb: Optional[float] = None


def create_input_file(input_length: int, batch_size: int, output_length: int) -> str:
    """Create input JSON for specified configuration."""

    # Generate prompt of approximately input_length tokens
    # (rough estimate: 1 word ~= 1.3 tokens)
    words_needed = int(input_length / 1.3)
    prompt = " ".join(["word"] * words_needed)

    input_data = {
        "batch_size": batch_size,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_generate_length": output_length,
        "requests": [
            {
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
        ] * batch_size
    }

    filename = f"benchmark_input_{input_length}_{batch_size}.json"
    with open(filename, "w") as f:
        json.dump(input_data, f, indent=2)

    return filename


def run_inference(engine_dir: str, input_file: str, output_file: str) -> Dict:
    """Run inference and return timing results."""

    cmd = [
        "./build/examples/llm/llm_inference",
        "--engineDir", engine_dir,
        "--inputFile", input_file,
        "--outputFile", output_file
    ]

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end = time.perf_counter()

    if result.returncode != 0:
        raise RuntimeError(f"Inference failed: {result.stderr}")

    # Parse output for detailed metrics
    with open(output_file, "r") as f:
        output_data = json.load(f)

    return {
        "total_time_ms": (end - start) * 1000,
        "responses": output_data.get("responses", [])
    }


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())
    except:
        return 0.0


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run a complete benchmark for a configuration."""

    print(f"\n{'='*60}")
    print(f"Benchmarking: {config.name}")
    print(f"  Engine: {config.engine_dir}")
    print(f"  Batch: {config.batch_size}, Input: {config.input_length}, Output: {config.output_length}")
    print(f"{'='*60}")

    # Create input file
    input_file = create_input_file(
        config.input_length,
        config.batch_size,
        config.output_length
    )
    output_file = f"benchmark_output_{config.name}.json"

    # Warmup runs
    print(f"Warmup ({config.warmup_runs} runs)...")
    for _ in range(config.warmup_runs):
        run_inference(config.engine_dir, input_file, output_file)

    # Benchmark runs
    print(f"Benchmarking ({config.num_runs} runs)...")
    results = []

    for i in range(config.num_runs):
        memory_before = get_gpu_memory_mb()
        result = run_inference(config.engine_dir, input_file, output_file)
        memory_after = get_gpu_memory_mb()

        # Extract per-response metrics
        if result["responses"]:
            resp = result["responses"][0]
            results.append({
                "prefill_ms": resp.get("prefill_time_ms", 0),
                "decode_ms": resp.get("decode_time_ms", 0),
                "tokens_per_sec": resp.get("tokens_per_second", 0),
                "tokens_generated": resp.get("tokens_generated", 0),
                "memory_mb": memory_after
            })
            print(f"  Run {i+1}: {resp.get('tokens_per_second', 0):.1f} tok/s")

    # Aggregate results
    if results:
        return BenchmarkResult(
            config_name=config.name,
            prefill_latency_ms=statistics.mean(r["prefill_ms"] for r in results),
            decode_latency_ms=statistics.mean(r["decode_ms"] for r in results),
            tokens_per_second=statistics.mean(r["tokens_per_sec"] for r in results),
            first_token_latency_ms=statistics.mean(r["prefill_ms"] for r in results),
            total_tokens=results[0]["tokens_generated"],
            memory_mb=statistics.mean(r["memory_mb"] for r in results)
        )

    return None


def print_comparison_table(results: List[BenchmarkResult]):
    """Print a comparison table of all results."""

    print("\n" + "="*80)
    print("BENCHMARK COMPARISON")
    print("="*80)

    # Header
    print(f"{'Config':<20} {'Prefill (ms)':<12} {'Decode (ms)':<12} "
          f"{'Tok/s':<10} {'Memory (MB)':<12}")
    print("-"*80)

    # Results
    for r in results:
        print(f"{r.config_name:<20} {r.prefill_latency_ms:<12.1f} "
              f"{r.decode_latency_ms:<12.1f} {r.tokens_per_second:<10.1f} "
              f"{r.memory_mb:<12.0f}")

    print("="*80)


def main():
    """Main benchmark entry point."""

    # Define configurations to benchmark
    configs = [
        BenchmarkConfig(
            name="fp8_short",
            engine_dir="./engines/fp8",
            input_length=64,
            output_length=64
        ),
        BenchmarkConfig(
            name="fp8_medium",
            engine_dir="./engines/fp8",
            input_length=256,
            output_length=128
        ),
        BenchmarkConfig(
            name="fp8_long",
            engine_dir="./engines/fp8",
            input_length=512,
            output_length=256
        ),
    ]

    # Check if INT4 engine exists
    if os.path.exists("./engines/int4"):
        configs.extend([
            BenchmarkConfig(
                name="int4_short",
                engine_dir="./engines/int4",
                input_length=64,
                output_length=64
            ),
            BenchmarkConfig(
                name="int4_medium",
                engine_dir="./engines/int4",
                input_length=256,
                output_length=128
            ),
        ])

    # Run benchmarks
    results = []
    for config in configs:
        try:
            result = run_benchmark(config)
            if result:
                results.append(result)
        except Exception as e:
            print(f"ERROR in {config.name}: {e}")

    # Print comparison
    if results:
        print_comparison_table(results)

        # Save results
        with open("benchmark_results.json", "w") as f:
            json.dump([{
                "name": r.config_name,
                "prefill_ms": r.prefill_latency_ms,
                "decode_ms": r.decode_latency_ms,
                "tokens_per_second": r.tokens_per_second,
                "memory_mb": r.memory_mb
            } for r in results], f, indent=2)
        print("\nResults saved to benchmark_results.json")


if __name__ == "__main__":
    main()
```

### Step 2: Prepare Multiple Engine Configurations

```bash
#!/bin/bash
# prepare_engines.sh - Build engines with different configurations

MODEL="Qwen/Qwen2.5-3B-Instruct"

# FP8 configuration
echo "Building FP8 engine..."
tensorrt-edgellm-quantize-llm \
    --model_dir $MODEL \
    --output_dir ./models/fp8 \
    --quantization fp8

tensorrt-edgellm-export-llm \
    --model_dir ./models/fp8 \
    --output_dir ./onnx/fp8

./build/examples/llm/llm_build \
    --onnxDir ./onnx/fp8 \
    --engineDir ./engines/fp8 \
    --maxBatchSize 4 \
    --maxKVCacheCapacity 2048

# INT4 configuration
echo "Building INT4 engine..."
tensorrt-edgellm-quantize-llm \
    --model_dir $MODEL \
    --output_dir ./models/int4 \
    --quantization int4_awq

tensorrt-edgellm-export-llm \
    --model_dir ./models/int4 \
    --output_dir ./onnx/int4

./build/examples/llm/llm_build \
    --onnxDir ./onnx/int4 \
    --engineDir ./engines/int4 \
    --maxBatchSize 4 \
    --maxKVCacheCapacity 4096  # INT4 uses less memory, can have larger cache

echo "Done! Engines ready in ./engines/"
```

### Step 3: Run Benchmarks

```bash
chmod +x prepare_engines.sh benchmark.py
./prepare_engines.sh
python benchmark.py
```

### Step 4: Memory Profiling

Create `memory_profile.py`:
```python
#!/usr/bin/env python3
"""
Memory profiling for TensorRT-Edge-LLM.

Tracks GPU memory usage during inference phases.
"""

import subprocess
import time
import threading
import json


class MemoryProfiler:
    """Continuous GPU memory sampling."""

    def __init__(self, sample_interval_ms: int = 100):
        self.sample_interval = sample_interval_ms / 1000
        self.samples = []
        self.running = False
        self.thread = None

    def _sample_memory(self):
        """Sample GPU memory in background thread."""
        while self.running:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True
                )
                memory_mb = float(result.stdout.strip())
                self.samples.append({
                    "timestamp_ms": time.perf_counter() * 1000,
                    "memory_mb": memory_mb
                })
            except:
                pass
            time.sleep(self.sample_interval)

    def start(self):
        """Start memory profiling."""
        self.samples = []
        self.running = True
        self.thread = threading.Thread(target=self._sample_memory)
        self.thread.start()

    def stop(self) -> dict:
        """Stop profiling and return results."""
        self.running = False
        if self.thread:
            self.thread.join()

        if not self.samples:
            return {}

        memories = [s["memory_mb"] for s in self.samples]
        return {
            "min_memory_mb": min(memories),
            "max_memory_mb": max(memories),
            "avg_memory_mb": sum(memories) / len(memories),
            "samples": len(self.samples),
            "timeline": self.samples
        }


def profile_inference(engine_dir: str, input_length: int, output_length: int):
    """Profile memory during inference."""

    # Create input
    input_data = {
        "batch_size": 1,
        "temperature": 0.7,
        "max_generate_length": output_length,
        "requests": [{"messages": [{"role": "user", "content": "x " * input_length}]}]
    }
    with open("profile_input.json", "w") as f:
        json.dump(input_data, f)

    # Start profiler
    profiler = MemoryProfiler(sample_interval_ms=50)
    profiler.start()

    # Run inference
    subprocess.run([
        "./build/examples/llm/llm_inference",
        "--engineDir", engine_dir,
        "--inputFile", "profile_input.json",
        "--outputFile", "profile_output.json"
    ], capture_output=True)

    # Stop and get results
    results = profiler.stop()

    print(f"\nMemory Profile for input={input_length}, output={output_length}")
    print(f"  Min: {results['min_memory_mb']:.0f} MB")
    print(f"  Max: {results['max_memory_mb']:.0f} MB")
    print(f"  Avg: {results['avg_memory_mb']:.0f} MB")
    print(f"  Peak-Base: {results['max_memory_mb'] - results['min_memory_mb']:.0f} MB")

    return results


if __name__ == "__main__":
    # Profile different configurations
    for input_len in [64, 256, 512, 1024]:
        profile_inference("./engines/fp8", input_len, 128)
```

### Step 5: Generate Report

Create `generate_report.py`:
```python
#!/usr/bin/env python3
"""Generate markdown benchmark report."""

import json
from datetime import datetime


def generate_report(results_file: str, output_file: str):
    """Generate markdown report from benchmark results."""

    with open(results_file, "r") as f:
        results = json.load(f)

    report = f"""# TensorRT-Edge-LLM Benchmark Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary

| Configuration | Prefill (ms) | Decode (ms) | Tokens/sec | Memory (MB) |
|---------------|--------------|-------------|------------|-------------|
"""

    for r in results:
        report += f"| {r['name']} | {r['prefill_ms']:.1f} | {r['decode_ms']:.1f} | "
        report += f"{r['tokens_per_second']:.1f} | {r['memory_mb']:.0f} |\n"

    report += """
## Analysis

### Quantization Impact

"""

    # Compare FP8 vs INT4 if both exist
    fp8_results = [r for r in results if "fp8" in r["name"]]
    int4_results = [r for r in results if "int4" in r["name"]]

    if fp8_results and int4_results:
        fp8_tps = fp8_results[0]["tokens_per_second"]
        int4_tps = int4_results[0]["tokens_per_second"]
        speedup = int4_tps / fp8_tps

        fp8_mem = fp8_results[0]["memory_mb"]
        int4_mem = int4_results[0]["memory_mb"]
        mem_saving = (fp8_mem - int4_mem) / fp8_mem * 100

        report += f"""
- INT4 vs FP8 throughput: {speedup:.2f}x
- INT4 memory savings: {mem_saving:.1f}%
"""

    report += """
## Recommendations

Based on the benchmark results:

1. **For latency-critical applications**: Use FP8 quantization
2. **For memory-constrained devices**: Use INT4 quantization
3. **For long context**: Reduce batch size or use INT4

## Hardware Info

Run `nvidia-smi` for current hardware details.
"""

    with open(output_file, "w") as f:
        f.write(report)

    print(f"Report saved to {output_file}")


if __name__ == "__main__":
    generate_report("benchmark_results.json", "BENCHMARK_REPORT.md")
```

## Example Output

```
============================================================
BENCHMARK COMPARISON
============================================================
Config               Prefill (ms)  Decode (ms)   Tok/s      Memory (MB)
--------------------------------------------------------------------------------
fp8_short            45.2          312.5         40.9       8234
fp8_medium           89.4          625.1         40.5       8456
fp8_long             156.8         1250.3        40.2       8892
int4_short           38.1          287.2         44.5       5123
int4_medium          72.3          574.8         44.1       5345
============================================================
```

## Exercises

### Exercise 1: Batch Size Scaling

Add configurations for different batch sizes:
```python
for batch_size in [1, 2, 4]:
    configs.append(BenchmarkConfig(
        name=f"fp8_batch{batch_size}",
        engine_dir="./engines/fp8",
        batch_size=batch_size,
        input_length=128,
        output_length=128
    ))
```

### Exercise 2: Context Length Analysis

Profile memory vs context length:
```python
for context in [512, 1024, 2048, 4096]:
    profile_inference("./engines/fp8", context, 128)
```

### Exercise 3: CUDA Graph Impact

Compare with/without CUDA graphs by modifying engine config.

## What You Learned

- ✅ How to systematically benchmark inference
- ✅ How to profile memory usage
- ✅ How to compare quantization schemes
- ✅ How to generate performance reports

## Next Steps

- Integrate benchmarks into CI/CD
- Add EAGLE speculative decoding comparison
- Profile specific kernel performance with Nsight
