#!/bin/bash
# Task 2: Setup script for Azure ML compute instance
# Run this after opening Terminal on the compute instance

set -e

echo "=== Installing Python dependencies ==="
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.49.0 accelerate bitsandbytes --upgrade
pip install pillow requests matplotlib
pip install peft datasets  # for later fine-tuning
pip install auto-gptq      # for later quantization
pip install qwen-vl-utils   # for Qwen2.5-VL image processing
pip install timm --upgrade  # for Florence-2 vision backbone
pip install backoff         # for Phi-4-multimodal

# flash-attn needs CUDA toolkit headers in the path
echo "=== Installing flash-attn ==="
export CUDA_HOME=$(dirname $(dirname $(which nvcc 2>/dev/null || echo "/usr/local/cuda/bin/nvcc")))
if [ ! -d "$CUDA_HOME/include" ]; then
    # Find CUDA installation
    for d in /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-12.9 /usr/local/cuda-12.8; do
        if [ -d "$d/include" ]; then
            export CUDA_HOME=$d
            break
        fi
    done
fi
echo "CUDA_HOME=$CUDA_HOME"
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install ninja for faster builds
pip install ninja

# Install flash-attn
pip install flash-attn --no-build-isolation

echo ""
echo "=== Downloading models ==="

# Florence-2-large (0.77B) — ~1.5 GB
echo "[1/4] Downloading Florence-2-large..."
python -c "
from transformers import AutoModelForCausalLM, AutoProcessor
AutoProcessor.from_pretrained('microsoft/Florence-2-large-ft', trust_remote_code=True)
AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large-ft', trust_remote_code=True)
print('  Florence-2-large done.')
"

# Qwen2.5-VL-3B — ~7 GB
echo "[2/4] Downloading Qwen2.5-VL-3B..."
python -c "
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')
Qwen2_5_VLForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')
print('  Qwen2.5-VL-3B done.')
"

# Qwen2.5-VL-7B — ~14 GB
echo "[3/4] Downloading Qwen2.5-VL-7B..."
python -c "
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')
Qwen2_5_VLForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')
print('  Qwen2.5-VL-7B done.')
"

# Phi-4-multimodal — ~12 GB
echo "[4/4] Downloading Phi-4-multimodal..."
python -c "
from transformers import AutoModelForCausalLM, AutoProcessor
AutoProcessor.from_pretrained('microsoft/Phi-4-multimodal-instruct', trust_remote_code=True)
AutoModelForCausalLM.from_pretrained('microsoft/Phi-4-multimodal-instruct', trust_remote_code=True)
print('  Phi-4-multimodal done.')
"

echo ""
echo "=== Setup complete. All 4 models downloaded. ==="
echo "Run: python inference_all_models.py <image_path>"
echo "Or open inference_notebook.ipynb in JupyterLab"
