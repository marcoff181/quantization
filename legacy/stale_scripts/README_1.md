# Image Generator Script

This project provides two scripts for generating images:

1.  `txt2img.py`: **Text-to-Image** (txt2img)
2.  `img2img.py`: **Image-to-Image** (img2img)

Support for various models (SDXL, SD 3.5, Flux.1) and quantization levels.

## Features

-   **Scripts**: Separated `txt2img` and `img2img` scripts.
-   **Models**:
    -   `sdxl`: Stable Diffusion XL Base 1.0
    -   `sd2`: Stable Diffusion 2.1
    -   `sd35`: Stable Diffusion 3.5 Medium
    -   `flux`: Flux.1 Dev
    -   `sd15`: Stable Diffusion 1.5
-   **Quantization**: `fp16` (Half Precision), `fp8` (8-bit), `fp4` (4-bit).

## Installation

Ensure you have the required dependencies installed:

```bash
pip install torch torchvision diffusers transformers accelerate bitsandbytes scipy numpy Pillow opencv-python sentencepiece protobuf
```

## Usage

Run the script from the command line.

### Text-to-Image (txt2img)

Generate images from a text prompt.

```bash
python txt2img.py \
    --prompt "A futuristic cyberpunk city with neon lights" \
    --models sdxl sd35 flux sd3 sd15 \
    --quantization fp16 fp8 fp4 \
    --output_dir output/txt2img
```


### Image-to-Image (img2img)

Use `img2img.py` for image-to-image generation.


```bash
python img2img.py \
    --input_dir path/to/input_images \
    --prompt "oil painting, van gogh style, thick brushstrokes" \
    --models sd15 sd3 sd35 \
    --quantization fp16 fp8 fp4 \
    --strength 0.6 \
    --output_dir output/img2img
```


### Arguments

| Argument | Script | Description | Default |
| :--- | :--- | :--- | :--- |
| `--prompt` | Both | Text prompt for generation | "" |
| `--models` | Both | List of models to use. `txt2img.py`: `sdxl`, `flux`, `sd3`, `sd35`, `sd15`. `img2img.py`: `sd15`, `sd3`, `sd35`. | `sdxl`/`sd15` |
| `--quantization` | Both | List of quantization levels | `fp16` |
| `--output_dir` | Both | Directory to save generated images | `output` |
| `--input_dir` | `img2img.py` OBS: Required | Directory containing input images | - |
| `--strength` | `img2img.py` | Denoising strength (0.0 to 1.0) | 0.3 |
| `--steps` | Both | Number of inference steps | 60 |
| `--guidance` | Both | Guidance scale | 3.5 |
| `--device` | Both | Device to use | `cuda`, `mps`, `cpu` |
| `--seed` | Both | Fixed seed for reproducibility | 123 |


## Directory Structure

-   `txt2img.py`: Text-to-Image script.
-   `img2img.py`: Image-to-Image script.

## Debug Scripts

Use these scripts to verify functionality:

**Text-to-Image Debug:**
```bash
python txt2img.py --mode txt2img --prompt "photo of an english setter" --models sdxl sd35 flux sd3 --quantization fp16 fp8 fp4 --output_dir output/txt2img
```

**Image-to-Image Debug:**
```bash
python img2img.py \
    --input_dir demo_images/ \
    --prompt "oil painting, van gogh style, thick brushstrokes" \
    --models sd15 sd3 sd35 \
    --quantization fp16 fp8 fp4 \
    --strength 0.6 \
    --output_dir output/img2img
```

