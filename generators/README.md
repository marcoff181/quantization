# quantization
This project aims to understand the impact of  the quantization of image generation models on the performance of common fake image detectors.

# usage
- run `hf auth login` and insert your hugging face token when prompted to be able to use the models
- if GPU 0 is busy (check with `nvidia-smi`) you can run on GPU 1 with `CUDA_VISIBLE_DEVICES=1 `

for txt2img run with 10 of the default prompts:
```
python txt2img.py --prompts 10 --output_dir /media/SSD_4TB/crispy_storage/
```

for now see README_1 for the rest

## txt2img workflow

```bash
python txt2img.py --models sd15 \
  --prompts_file 15_no_item_and_15_with_controversial.txt \
  --quantization fp16 \
  --output_dir /media/SSD_4TB/crispy_storage/to_modify
```

### Sharded single-model mode (for models that do not fit one GPU)

Use `CUDA_VISIBLE_DEVICES` plus `--device_map` to shard one pipeline across multiple GPUs.

```bash
CUDA_VISIBLE_DEVICES=0,1 python txt2img.py \
  --models flux \
  --quantization fp16 \
  --device_map balanced \
  --dtype fp16 \
  --max_memory 0=20GiB 1=20GiB \
  --prompt "a cinematic photo of a red fox in snow" \
  --output_dir /media/SSD_4TB/crispy_storage/elia/
```

Notes:
- In sharded mode, the script does not call `.to(...)` and does not enable CPU offload hooks.
- `hf_device_map` is printed after load so you can verify placement.
- Sharded mode currently targets: `flux`, `flux2`, `sd3`, `sd35`.

### Mixed quantization in one command (fp16 + fp8 + fp4)

You can now run all quantization levels in a single invocation. In sharded mode:
- `fp16` stays sharded.
- `fp8`/`fp4` are controlled by `--sharded_quant_fallback`.

Default behavior is `reroute`, which runs `fp8`/`fp4` as non-sharded single-device jobs.

```bash
CUDA_VISIBLE_DEVICES=0,1 python txt2img.py \
  --models flux \
  --quantization fp16 fp8 fp4 \
  --device_map balanced \
  --dtype fp16 \
  --sharded_quant_fallback reroute \
  --max_memory 0=20GiB 1=20GiB \
  --prompt "a cinematic photo of a red fox in snow" \
  --output_dir /media/SSD_4TB/crispy_storage/elia/
```

Fallback modes:
- `--sharded_quant_fallback reroute` (default): reroute `fp8`/`fp4` to non-sharded execution.
- `--sharded_quant_fallback skip`: skip `fp8`/`fp4` when `--device_map` is enabled.
- `--sharded_quant_fallback error`: fail fast if `fp8`/`fp4` is requested with `--device_map`.

## Inpainting workflow

The inpainting workflow is split into two separate scripts. You can use a single CSV file with per-image prompts for both mask generation and inpainting.

### CSV file

Create a single CSV file (e.g., `masks_auto.csv`) with three columns:

```
filename,mask_prompt,inpaint_prompt
00000,face,professional headshot, studio lighting, high quality
00001,face,corporate portrait, neutral background
00002,background,landscape painting, turquoise sky
```

- **Column 1** (`filename`): Image filename or basename
- **Column 2** (`mask_prompt`): What to mask (used by `generate_masks.py`)
- **Column 3** (`inpaint_prompt`): How to inpaint the masked region (used by `inpaint.py`)

Images not in the CSV will use default prompts from CLI flags.

### 1. Generate masks with CLIPSeg (`generate_masks.py`)

```bash
python generate_masks.py \
   --input_dir /tmp/demo_img/ \
   --output_dir out/inpaint/masks \
   --mask_prompts_file masks_auto.csv \
   --max_images 3 \
   --overwrite
```

**Editing masks** (optional): The generated `.png` masks can be manually edited in any image editor (white = inpaint, black = preserve).

### 2. Run inpainting (`inpaint.py`)

```bash
# Using CLIPSeg masks from step 1
python inpaint.py  \
  --input_dir /tmp/demo_img/ \
  --mask_dir ./out/inpaint/masks \
  --prompts_file masks_auto.csv \
  --strength 0.75 \
  --guidance 8.0  \
  --steps 30  \
  --models sd15  \
  --preserve_aspect_ratio \
  --max_images 2 \
  --quantization fp16 fp8 fp4 \
  --models sd3

# Using geometric masks (no pre-generation needed)
python inpaint.py \
  --input_dir /tmp/demo_img/ \
  --mask_type center_circle \
  --mask_size_ratio 0.3 \
  --prompt "oil painting" \
  --models sd15 \
  --strength 0.75 \
  --guidance 8.0  \
  --steps 30  \
  --output_dir out/inpaint
```

**Inpainting parameters**:
- `--strength 0.75`: Lower = more faithful to original
- `--guidance 8.0`: Lower = less prompt influence
- `--steps 30`: Inference steps
- `--preserve_aspect_ratio`: Enable smart padding to preserve image aspect ratio instead of resizing to square
- `--max_dimension 512`: Maximum dimension when preserving aspect ratio (longer side won't exceed this)
- `--pad_color "128,128,128"`: RGB padding color as 'R,G,B' when preserving aspect ratio (default: neutral gray)