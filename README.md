# ComfyServer

ComfyServer is a custom KServe serving runtime for deploying ComfyUI workflows as inference services. It bridges ComfyUI's execution model with KServe's Open Inference Protocol (OIP), enabling you to serve ComfyUI image generation workflows via REST/gRPC APIs.

## Features

- **Universal Workflow Support**: Execute arbitrary ComfyUI workflows exported as JSON
- **Custom Nodes**: Support for ComfyUI custom nodes and extensions
- **Optimized Caching**: Preloads model weights at startup for minimal per-request latency
- **Protocol Compliant**: Implements KServe Open Inference Protocol V2
- **Swagger UI**: Built-in API documentation when `--enable_docs_url` is enabled

## Quick Start

### Prerequisites

- Docker with NVIDIA GPU support (`nvidia-container-toolkit`)
- NVIDIA GPU with CUDA support
- Models downloaded (see [Model Setup](#model-setup))

### 1. Build the Docker Image

```bash
docker build -f comfy.Dockerfile -t comfyserver .
```

For extended custom node support:
```bash
docker build -f comfy_extra_custom_nodes.Dockerfile -t comfyserver:extra .
```

#### Prebuilt Image (GitHub Packages)

Alternatively, you can pull the prebuilt image directly from GitHub Container Registry:

```bash
docker pull ghcr.io/mjzlotr/comfyserver:latest
```

See all available tags at: [GitHub Packages - ComfyServer](https://github.com/MJZLOTR/ComfyServer/pkgs/container/comfyserver)

### 2. Model Setup

ComfyServer requires model files to be mounted into the container. Follow these steps:

#### Directory Structure

Create a directory structure for your models:

```
models/
├── checkpoints/           # Main diffusion models (.safetensors, .ckpt)
├── vae/                   # VAE models
├── clip/                  # CLIP text encoders
├── loras/                 # LoRA adapters
├── controlnet/            # ControlNet models
├── upscale_models/        # Upscaling models
└── embeddings/            # Textual inversion embeddings
```

#### Download Models

Download the required model files for your workflow from [Hugging Face](https://huggingface.co/), [Civitai](https://civitai.com/), or other model repositories. Place them in the appropriate directories based on their type.

> **Note**: Some models require authentication with Hugging Face:
> ```bash
> pip install huggingface_hub
> huggingface-cli login
> ```

#### Create Config File

Create a `config.yaml` file to map model paths (see [models_path_config.yaml](./testing/models_path_config.yaml)):

```yaml
comfyui:
    base_path: /mnt/models/
    is_default: true
    checkpoints: models/checkpoints/
    clip: models/clip/
    clip_vision: models/clip_vision/
    configs: models/configs/
    controlnet: models/controlnet/
    diffusion_models: models/diffusion_models/
    embeddings: models/embeddings/
    loras: models/loras/
    upscale_models: models/upscale_models/
    vae: models/vae/
```

### 3. Run the Container

> **Important**: You must specify:
> - `--workflow`: Path to your ComfyUI workflow JSON file (exported from ComfyUI)
> - `--model_name`: A name for your service, which becomes part of the inference URL: `/v2/models/<model_name>/infer`

```bash
docker run -it --rm \
  -e PORT=8080 \
  -p 8080:8080 \
  --gpus all \
  -v "/path/to/your/models-and-workflows":/mnt/models \
  comfyserver \
  --http_port 8080 \
  --enable_grpc false \
  --predictor_protocol v2 \
  --workers 1 \
  --max_threads 4 \
  --enable_docs_url true \
  --model_name "my-workflow" \
  --models_path_config /mnt/models/models_path_config.yaml \
  --workflow /mnt/models/workflow.json \
  --disable_save_nodes true \
  --enable_extra_builtin_nodes true \
  --enable_custom_nodes true \
  --enable_api_nodes false
```

#### Command-Line Arguments

| Argument | Description |
|----------|-------------|
| `--http_port` | HTTP server port (default: 8080) |
| `--enable_grpc` | Enable gRPC endpoint (default: true) |
| `--predictor_protocol` | Protocol version: `v1` or `v2` |
| `--workers` | Number of worker threads |
| `--max_threads` | Maximum thread pool size |
| `--enable_docs_url` | Enable Swagger UI at `/docs` |
| `--model_name` | Name for the inference service (used in URL: `/v2/models/<name>/infer`) |
| `--models_path_config` | Path to model directory config YAML |
| `--workflow` | Path to ComfyUI workflow JSON file |
| `--disable_save_nodes` | Disable file-saving nodes |
| `--enable_extra_builtin_nodes` | Load extra built-in ComfyUI nodes |
| `--enable_custom_nodes` | Load custom nodes from `custom_nodes/` |
| `--enable_api_nodes` | Enable API integration nodes |

### 4. API Documentation (Swagger UI)

When running with `--enable_docs_url true`, access the interactive API documentation at:

```
http://localhost:8080/docs
```

The Swagger UI allows you to:
- Inspect available API endpoints
- View request/response schemas
- Test inference requests directly from the browser

## Demo: Simple Workflow

This demo uses the simple Stable Diffusion 1.5 text-to-image workflow.

### Setup

1. **Prepare the Docker image** (choose one option):
   - **Option A: Use prebuilt image**
     ```bash
     docker pull ghcr.io/mjzlotr/comfyserver:latest
     ```
   - **Option B: Build locally**
     ```bash
     docker build -f comfy.Dockerfile -t comfyserver .
     ```

2. **Download the model** (if not already done):
   ```bash
   cd testing && ./download_models.sh && cd ..
   ```

3. **Start the server**:
   ```bash
   # Use 'ghcr.io/mjzlotr/comfyserver:latest' if using prebuilt, or 'comfyserver' if built locally
   docker run -it --rm \
     -e PORT=8080 \
     -p 8080:8080 \
     --gpus all \
     -v "$(pwd)/testing":/mnt/models \
     comfyserver \
     --http_port 8080 \
     --enable_grpc false \
     --predictor_protocol v2 \
     --workers 1 \
     --max_threads 4 \
     --enable_docs_url true \
     --model_name "simple-workflow" \
     --models_path_config /mnt/models/models_path_config.yaml \
     --workflow /mnt/models/testing/workflows/basic/simple_wf.json \
     --disable_save_nodes true \
     --enable_extra_builtin_nodes true \
     --enable_custom_nodes true \
     --enable_api_nodes false
   ```


4. **Wait for the server to be ready** (model loading may take a few minutes)

### Generate an Image

Use the provided test script:

```bash
./testing/simple_wf.sh
```

Or send a request manually with curl:

```bash
curl -X POST http://localhost:8080/v2/models/simple-workflow/infer \
  -H "Content-Type: application/json" \
  -d '{
    "id": "1",
    "inputs": [
      {"name": "3_steps", "datatype": "INT32", "shape": [1], "data": [20]},
      {"name": "6_text", "datatype": "BYTES", "shape": [1], "data": ["a beautiful sunset over mountains"]}
    ],
    "outputs": [
      {
        "name": "8_0_IMAGE",
        "datatype": "FP32",
        "shape": [-1],
        "parameters": {
          "binary_data": false,
          "to_base64": true
        }
      }
    ]
  }' | jq -r '.outputs[0].data[0]' | base64 -d > generated_image.png
```

### Request Format

The inference request follows the Open Inference Protocol V2 format. Input/output names follow the pattern:
- **Inputs**: `{node_id}_{input_name}` (e.g., `6_text` for node 6's text input)
- **Outputs**: `{node_id}_{output_index}_{type}` (e.g., `8_0_IMAGE` for node 8's first IMAGE output)

## Demo: Image Input with Base64

This demo shows how to send images in your inference requests using base64 encoding. This is useful for image-to-image workflows, image editing, and other use cases that require input images.

See the [OmniGen2 Examples](https://comfyanonymous.github.io/ComfyUI_examples/omnigen/) for workflow references.

### Setup

1. **Prepare the Docker image** and **download the OmniGen2 model** following the model setup instructions above.

2. **Start the server** with the OmniGen2 image editing workflow:
   ```bash
   docker run -it --rm \
     -e PORT=8080 \
     -p 8080:8080 \
     --gpus all \
     -v "$(pwd)/testing":/mnt/models \
     comfyserver \
     --http_port 8080 \
     --enable_grpc false \
     --predictor_protocol v2 \
     --workers 1 \
     --max_threads 4 \
     --enable_docs_url true \
     --model_name "image_omnigen2_image_edit" \
     --models_path_config /mnt/models/models_path_config.yaml \
     --workflow /mnt/models/testing/workflows/etc/image_omnigen2_image_edit.json \
     --disable_save_nodes true \
     --enable_extra_builtin_nodes true \
     --enable_custom_nodes true \
     --enable_api_nodes false
   ```

### Send an Image via Base64

Use the provided test script:

```bash
./testing/omni_edit_request.sh
```

The script demonstrates how to:

1. **Encode an image to base64**:
   ```bash
   IMAGE_B64=$(base64 -w 0 "./your_image.png")
   ```

2. **Include the base64 image in the request** with `as_base64: true` parameter:
   ```bash
   curl -X POST http://localhost:8080/v2/models/image_omnigen2_image_edit/infer \
     -H "Content-Type: application/json" \
     -d '{
       "id": "1",
       "inputs": [
         {
           "name": "6_text",
           "datatype": "BYTES",
           "shape": [1],
           "data": ["your prompt describing the edit"]
         },
         {
           "name": "16_image",
           "datatype": "BYTES",
           "shape": [1],
           "data": ["'"$IMAGE_B64"'"],
           "parameters": {
             "as_base64": true
           }
         }
       ],
       "outputs": [
         {
           "name": "8_0_IMAGE",
           "datatype": "FP32",
           "shape": [-1],
           "parameters": {
             "binary_data": false,
             "to_base64": true
           }
         }
       ]
     }' | jq -r '.outputs[0].data[0]' | base64 -d > generated_image.png
   ```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `as_base64: true` | Indicates the input data is base64-encoded |
| `to_base64: true` | Returns output image as base64 (in response) |

## Workflow Templates

The `testing/workflows/` directory contains example workflows. All workflows have been tested in ComfyUI before being exported for use with ComfyServer.

| Workflow | Description | ComfyUI Template |
|----------|-------------|------------------|
| `basic/simple_wf.json` | Basic text-to-image with SD 1.5 | — |
| `basic/image2image.json` | Image-to-image transformation | [Image-to-Image](https://comfyanonymous.github.io/ComfyUI_examples/img2img/) |
| `basic/lora.json` | LoRA model usage | [LoRA Example](https://comfyanonymous.github.io/ComfyUI_examples/lora/) |
| `basic/inpaint_example.json` | Inpainting workflow | [Inpaint Example](https://comfyanonymous.github.io/ComfyUI_examples/inpaint/) |
| `flux/` | FLUX model workflows | [FLUX Examples](https://comfyanonymous.github.io/ComfyUI_examples/flux/) |
| `etc/image_qwen_image.json` | Qwen2.5-VL text-to-image | [Qwen Image Gen](https://comfyanonymous.github.io/ComfyUI_examples/qwen_image/) |
| `etc/image_omnigen2_t2i.json` | OmniGen2 text-to-image | [OmniGen2 Examples](https://comfyanonymous.github.io/ComfyUI_examples/omnigen/) |
| `etc/image_omnigen2_image_edit.json` | OmniGen2 image editing with reference | [OmniGen2 Examples](https://comfyanonymous.github.io/ComfyUI_examples/omnigen/) |

## Architecture

ComfyServer operates as a three-layer adapter:

1. **Protocol Server**: Handles OIP V2 REST/gRPC requests
2. **Data Adapter**: Converts between OIP types and ComfyUI types
3. **Workflow Execution**: Orchestrates ComfyUI node execution

```
┌─────────────────┐
│   REST/gRPC     │  ← Client requests
├─────────────────┤
│ Protocol Server │  ← KServe OIP V2
├─────────────────┤
│  Data Adapter   │  ← Type conversion
├─────────────────┤
│Workflow Executor│  ← ComfyUI nodes
└─────────────────┘
```

## Local Development

For development without Docker:

### Prerequisites

- Python 3.10+
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) cloned locally

### Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Set up environment variables
export COMFY_PATH=/path/to/ComfyUI
export PYTHONPATH="${COMFY_PATH}:$(pwd)"

# Install ComfyUI dependencies (required)
pip install -r ${COMFY_PATH}/requirements.txt

# Install ComfyServer dependencies
pip install -r requirements.txt
```

### Run the Server

```bash
python -m comfyserver \
  --workflow path/to/workflow.json \
  --models_path_config path/to/config.yaml \
  --predictor_protocol v2 \
  --enable_docs_url true
```

