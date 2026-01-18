# JetPack 7 Setup on Jetson Thor

This document covers the complete setup process for JetPack 7 on Jetson Thor, including dependency resolution, jtop installation, CUDA setup, and PyTorch installation for ML workloads.

## Overview

- **Platform**: Jetson Thor
- **JetPack**: 7.x (L4T R38.x)
- **CUDA**: 13.0
- **Ubuntu**: 24.04
- **Python**: 3.12

---

## 1. JetPack 7 Installation

### Resolving Container Package Version Skew

During `apt install nvidia-jetpack-7`, you may encounter unmet dependencies in the NVIDIA container stack. JetPack 7 (r38.4) strictly requires container packages version `1.18.0-1`, but your system may have newer versions (`1.18.1-1`) installed.

**Diagnose the issue:**

```bash
apt-cache policy nvidia-container-toolkit
```

**Fix by downgrading and pinning:**

```bash
# Downgrade to required version
sudo apt install nvidia-container-toolkit=1.18.0-1 \
    nvidia-container-toolkit-base=1.18.0-1

# Hold packages to prevent re-upgrade
sudo apt-mark hold nvidia-container-toolkit nvidia-container-toolkit-base

# Now install JetPack
sudo apt update
sudo apt install nvidia-jetpack
```

---

## 2. CUDA Installation

**Use ONLY the JetPack CUDA meta-package:**

```bash
sudo apt update
sudo apt install nvidia-cuda-dev
```

**DO NOT install Ubuntu's cuda toolkit:**

```bash
# AVOID THIS - breaks Jetson-specific drivers
# sudo apt install nvidia-cuda-toolkit
```

The `nvidia-cuda-toolkit` package is maintained by Ubuntu, not NVIDIA Jetson, and installs an incompatible CUDA stack.

---

## 3. jtop Installation (jetson-stats)

### The Problem

Standard pip installation fails due to PEP 668, which marks system Python as externally managed on JetPack 7 / Ubuntu 24.04:

```bash
# This will fail
sudo pip3 install -U jetson-stats
```

Jetson Thor requires a newer jtop version with Thor support than what APT provides.

### Recommended Method (Official Script)

```bash
# Download the install script
wget https://raw.githubusercontent.com/rbonghi/jetson_stats/master/scripts/install_jtop_torun_without_sudo.sh

# Make executable
chmod +x install_jtop_torun_without_sudo.sh

# Validate sudo credentials
sudo -v

# Run installer
./install_jtop_torun_without_sudo.sh

# Run jtop
jtop
# or
sudo jtop
```

### Alternative Method (Use with Caution)

This bypasses PEP 668 safeguards:

```bash
sudo pip install --break-system-packages -U git+https://github.com/rbonghi/jetson_stats.git
sudo jtop
```

---

## 4. PyTorch Installation for ML Workloads

PyTorch must be installed in a virtual environment on JetPack 7. Installing into system Python is intentionally blocked by PEP 668.

### Create Virtual Environment

```bash
cd ~/your-project

# Remove old venv if exists
rm -rf .venv

# Create venv with system site-packages access (for CUDA libs)
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
```

### Install PyTorch for JetPack 7

**Option 1: Jetson AI Lab Mirror**

```bash
pip install --no-cache-dir torch torchvision \
    --index-url https://pypi.jetson-ai-lab.dev/jp7/cu130/
```

**Option 2: NVIDIA Redist**

```bash
pip install --no-cache-dir torch \
    --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v71/pytorch/
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output:
```
PyTorch 2.x.x
CUDA available: True
Device: <Thor GPU name>
```

---

## 5. TensorRT-Edge-LLM Setup

After PyTorch is working with CUDA:

```bash
cd ~/tensorRT-Edge-LLM-Learn
source .venv/bin/activate

# Install dependencies (skip torch since its already installed)
pip install transformers==4.57.1 \
    nvidia-modelopt[torch]==0.39.0 \
    nvidia-modelopt[onnx]==0.39.0 \
    onnx==1.19.0 \
    datasets==4.0.0 \
    tqdm~=4.67.1 \
    numpy~=2.2.6 \
    peft~=0.18.0 \
    backoff~=2.2.1

# Install package without dependencies
pip install --no-deps -e .

# Test quantization
tensorrt-edgellm-quantize-llm \
    --model_dir Qwen2.5-0.5B-Instruct \
    --output_dir Qwen2.5-0.5B-Instruct-fp8 \
    --quantization fp8
```

---

## Key Takeaways

1. **JetPack version pinning**: JetPack requires exact package versions. Downgrade and hold conflicting packages.

2. **Use JetPack meta-packages**: Always use `nvidia-cuda-dev`, never Ubuntu's `nvidia-cuda-toolkit`.

3. **PEP 668 is intentional**: System Python is protected. Use virtual environments for ML frameworks.

4. **Thor-specific tooling**: Jetson Thor needs newer versions of tools like jtop. Use official install scripts.

5. **Virtual environments are mandatory**: PyTorch and ML frameworks must be installed in venvs, not system Python.

---

## Troubleshooting

### PyTorch shows CUDA unavailable

```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Verify venv has system site-packages
python -c "import sys; print(sys.path)"

# Reinstall with system site-packages
rm -rf .venv
python3 -m venv .venv --system-site-packages
```

### Container build failures

JetPack 7 container support is still maturing. If `jetson-containers build` fails, use direct pip installation in a venv instead.

### APT dependency errors

```bash
# Check what's holding packages
apt-cache policy <package-name>

# Force specific version
sudo apt install <package>=<version>

# Hold to prevent upgrades
sudo apt-mark hold <package>
```
