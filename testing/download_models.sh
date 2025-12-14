#!/bin/bash
# Download SD 1.5 model and create config for ComfyServer

set -e

# Create model directories
mkdir -p models/{checkpoints,vae,clip,clip_vision,loras,controlnet,upscale_models,embeddings,diffusion_models,configs}

# Download model if not present
MODEL="models/checkpoints/v1-5-pruned-emaonly.safetensors"
if [ ! -f "$MODEL" ]; then
    echo "Downloading Stable Diffusion 1.5..."
    wget -O "$MODEL" "https://huggingface.co/sd-legacy/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"
fi


echo "Done! Model: $MODEL"
