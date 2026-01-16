# Sample Projects

Hands-on projects to solidify your understanding of TensorRT-Edge-LLM.

## Projects Overview

| Project | Difficulty | Focus Area |
|---------|------------|------------|
| [hello_inference](hello_inference/) | Beginner | Complete pipeline walkthrough |
| [multi_adapter](multi_adapter/) | Intermediate | LoRA adapter management |
| [benchmark_suite](benchmark_suite/) | Advanced | Performance profiling |

## Prerequisites

Before starting any project:

```bash
# Verify TensorRT-Edge-LLM is installed
python -c "import tensorrt_edgellm; print('OK')"

# Verify CUDA is available
nvidia-smi

# Verify TensorRT
trtexec --version
```

## Quick Start

```bash
# Start with the simplest project
cd hello_inference
./run.sh
```

## Project Structure

Each project contains:
```
project_name/
├── README.md          # Project guide
├── run.sh             # One-click execution script
├── input.json         # Sample input
└── *.py / *.cpp       # Source code (if applicable)
```

## Learning Path Integration

These projects map to the tutorial levels:

| Tutorial Level | Recommended Projects |
|----------------|---------------------|
| After Level 1 | hello_inference |
| After Level 4 | multi_adapter |
| After Level 5 | benchmark_suite |
