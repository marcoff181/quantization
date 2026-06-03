# CrispyMcMarkInc — AI-Generated Image Detection Under Quantization

This repository studies how **model quantization** (fp16 / fp8 / fp4) of diffusion models affects **deepfake image detection**. It has two parallel tracks:

1. **Text-to-Image (txt2img)**: Generate fake images from text prompts using quantized models, then run multiple detectors.
2. **Image-to-Image (img2img)**: Generate subtle perturbations of real images using quantized models at low strength (0.3), then fine-tune **CLIP-D** to detect them.

The detection library is a git submodule: [Image-Deepfake-Detectors-Public-Library](https://github.com/grip-unina/ClipBased-SyntheticImageDetection) (GRIP-UNINA).

---

## Project Structure

| Directory / File | Purpose |
|---|---|
| `quantization/` | txt2img, inpainting generators + shared utilities |
| `original_img_to_img/` | img2img generator engine |
| `Image-Deepfake-Detectors-Public-Library/` | Detector library (submodule): CLIP-D, R50_TF, NPR, P2G, R50_nodown |
| `generate_images.py` | Batch runner for txt2img generation |
| `generate_images_img2img_splits.py` | Batch runner for img2img generation (split-aware) |
| `generate_img2img_manifests.py` | Builds CSV manifests for CLIP-D training |
| `prepare_img2img_splits.py` | Creates train/val/test text splits from real images |
| `crop_real_images.py` | Center-crops real images to 1024×1024 |
| `run_detection.py` | Runs CLIP-D + R50_TF + NPR on txt2img outputs |
| `run_detection_img2img.py` | Runs CLIP-D (default/patch/grid modes) on img2img outputs |
| `run_detection_on_real.py` | Runs all CLIP-D modes on real images for baseline |
| `analyze_detector_performance.py` | Analyzes txt2img detection results |
| `analyze_detector_performance_img2img.py` | Analyzes img2img detection results |
| `analyze_clipd_comparison_report.py` | Compares CLIP-D modes (ROC/PR curves, metrics) |
| `analyze_clipd_img2img_patch.py` | Patch/grid detection analysis |
| `temp_anal.py` | Real-image baseline analysis |
| `splits/` | Train/val/test split text files (image-level) |
| `manifests/` | CSV manifests (path + label) for CLIP-D training |
| `checkpoint/` | Fine-tuned CLIP-D weights (v1, v2) |
| `results/` | Test metrics and per-image predictions |

---

## Pipeline 1: Quantized txt2img Generation + Detection

### 1.1 Generate images

```bash
python generate_images.py \
  --models sd15 sd3 sd35 sdxl \
  --quantizations fp16 fp8 fp4 \
  --prompts 100 \
  --output_dir /media/CrispyMcMarkInc/fake-images/
```

This calls `quantization/txt2img.py` as subprocess for every model×quantization combination. Supported models: `sd15`, `sd3`, `sd35`, `sdxl`, `flux`, `z-image`, `pg25`. Prompts come from `quantization/prompts_filtered.txt` (5771 prompts).

### 1.2 Run detection

```bash
python run_detection.py \
  --input_dir /media/CrispyMcMarkInc/fake-images/ \
  --results_csv detector_final_results.csv \
  --detectors CLIP-D R50_TF NPR
```

Writes per-image predictions with confidence, detection mode, and elapsed time to CSV.

### 1.3 Analyze

```bash
python analyze_detector_performance.py
```

---

## Pipeline 2: Quantized img2img Generation + CLIP-D Fine-Tuning

This is the main experimental track. The goal is to detect subtle img2img edits produced by quantized diffusion models at low strength (0.3).

### 2.1 Dataset preparation

```bash
# Crop real images to 1024×1024
python crop_real_images.py \
  --input_dir /media/CrispyMcMarkInc/Real/ \
  --output_dir /media/CrispyMcMarkInc/Real_cropped_1024/

# Split into train/val/test (70/15/15) at the image level
python prepare_img2img_splits.py \
  --input_dir /media/CrispyMcMarkInc/Real_cropped_1024 \
  --output_dir /media/CrispyMcMarkInc/splits \
  --limit 500 \
  --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15
```

Real images come from the FORLAB dataset (first 1000 images).

### 2.2 Generate img2img fakes

```bash
python generate_images_img2img_splits.py \
  --split_dir /media/CrispyMcMarkInc/splits \
  --prefix img2img \
  --splits train val test \
  --output_root /media/CrispyMcMarkInc/fake-images-img2img \
  --models sd15 sd3 sd35 \
  --quantizations fp16 \
  --strength 0.3 --steps 30 --guidance 3.5
```

This calls `original_img_to_img/original_img2img.py` for each image. Each real image is perturbed through the diffusion model at strength 0.3 — subtle enough that the fake is visually nearly identical to the original.

### 2.3 Build training manifests

```bash
python generate_img2img_manifests.py \
  --split_dir /media/CrispyMcMarkInc/splits \
  --output_dir /media/CrispyMcMarkInc/manifests \
  --fake_root /media/CrispyMcMarkInc/fake-images-img2img
```

Outputs CSV files with columns `[path, label]` where label 0 = real, 1 = fake.

### 2.4 Fine-tune CLIP-D

```bash
python /media/CrispyMcMarkInc/Image-Deepfake-Detectors-Public-Library/detectors/CLIP-D/train.py \
  --name img2img_ft_fp16_v2 \
  --arch opencliplinearnextft_clipL14commonpool \
  --task train \
  --device cuda:0 \
  --csv_train /media/CrispyMcMarkInc/manifests/img2img_train.csv \
  --csv_val /media/CrispyMcMarkInc/manifests/img2img_val.csv \
  --csv_test /media/CrispyMcMarkInc/manifests/img2img_test.csv \
  --pretrained_weights /media/CrispyMcMarkInc/Image-Deepfake-Detectors-Public-Library/detectors/CLIP-D/checkpoint/pretrained/weights/best.pt \
  --batch_size 8 \
  --lr 5e-5 \
  --backbone_lr 1e-6 \
  --num_epoches 20 \
  --earlystop_epoch 3
```

Key details:
- **Architecture**: `OpenClipLinearFT` — a fine-tunable wrapper around OpenCLIP ViT-L/14 (CommonPool pretrained). Gradients flow through the backbone.
- **Two learning rates**: the linear head at 5e-5, the backbone at 1e-6 (10× smaller, preserving visual knowledge while adapting features).
- **Early stopping**: patience of 3 epochs; when validation balanced accuracy plateaus, LR is divided by 10 (min 1e-6).
- **Augmentation**: RandomResizedCrop (20% prob) + Random JPEG compression (50% prob), then resize to 224×224 + CLIP normalization.
- **Loss**: BCEWithLogitsLoss. **Optimizer**: Adam (β₁=0.9, weight decay=0.0).

### 2.5 Evaluate

```bash
python /media/CrispyMcMarkInc/Image-Deepfake-Detectors-Public-Library/detectors/CLIP-D/test.py \
  --name img2img_ft_fp16_v2 \
  --arch opencliplinearnextft_clipL14commonpool \
  --task test \
  --device cuda:0 \
  --csv_test /media/CrispyMcMarkInc/manifests/img2img_test.csv
```

### 2.6 Results (fine-tuned model on held-out test set)

| Metric | Value |
|---|---|
| TPR | 0.991 |
| TNR | 0.960 |
| Accuracy | 0.983 |
| AUC | 0.997 |

### 2.7 Why full-image fine-tuning, not patch/grid

The frozen-baseline CLIP-D (linear head only) plateaus at AUC ~0.65 on this task. The img2img edits at strength 0.3 are too subtle for generic CLIP features to separate linearly. Fine-tuning the backbone with a small learning rate lets the feature space adapt to the specific artifacts introduced by the quantized img2img pipeline.

Full-image analysis outperforms patch/grid inference because:
- CLIP-D is designed for **full images** resized to 224×224, not isolated crops
- Crops break global context, causing false positives on real images
- Aggregation (max/majority) amplifies outlier patch errors
- Img2img artifacts are visible in **global statistics** rather than local textures

---

## Pipeline 3: Inpainting (experimental)

```bash
# Generate CLIPSeg masks
python quantization/generate_masks.py \
  --input_dir /tmp/demo_img/ \
  --output_dir out/inpaint/masks \
  --mask_prompts_file quantization/new_masks_auto.csv

# Run inpainting
python quantization/inpaint.py \
  --input_dir /tmp/demo_img/ \
  --mask_dir out/inpaint/masks \
  --prompts_file quantization/new_masks_auto.csv \
  --models sd15 sd3 \
  --quantization fp16 fp8 fp4 \
  --strength 0.75 --guidance 8.0 --steps 30
```

---

## Detectors (submodule)

All detectors live in `Image-Deepfake-Detectors-Public-Library/detectors/`. The project primarily uses:

| Detector | Architecture | Weights |
|---|---|---|
| **CLIP-D** | OpenCLIP ViT-L/14 + linear head | `opencliplinearnext_clipL14commonpool` (frozen) or `opencliplinearnextft_clipL14commonpool` (fine-tuned) |
| **R50_TF** | ResNet-50 with three-filter | `nodown` arch |
| **NPR** | Noise Pattern Residual | — |
| **R50_nodown** | ResNet-50 without stride-2 downsampling | `res50nodown` arch |
| **P2G** | Pixel-to-Gram | — |

---

## Stale / Legacy Code

The following scripts exist but are **not part of the active pipeline**:

| File | Why it is stale |
|---|---|
| `move_images.py` | One-time FORLAB copy; hardcoded NAS path |
| `generate_images_img2img.py` | Early flat-folder batch runner; superseded by `generate_images_img2img_splits.py` |
| `quantization/img2img.py` | Duplicate img2img implementation; the active one is `original_img_to_img/original_img2img.py` |
| `quantization/components.py` | Debug script for listing model quantizable components |
| `analisi_risultati.py` | Italian-language aesthetic scoring (unrelated to deepfake detection) |
| `some_copy_with_permission_of_pietro/` | Parallel experiment on different paths with parameter grid search |
| `original_img_to_img/original_img2img_run.sh` | One-off shell example |
| `finetune_clipd_README.md` | Earlier fine-tuning guide (different arch name, no backbone_lr) |
| `quantization/README_1.md` | Superseded by `quantization/README.md` |
| `Image-Deepfake-Detectors-Public-Library/quantization_accuracy_report.md` | Empty template |

---

## Requirements

Two conda/virtual environments are expected under `/media/CrispyMcMarkInc/.venvs/`:

- **`detector`**: PyTorch 2.4, open-clip-torch, scikit-learn, pandas, matplotlib, seaborn
- **`quantization`**: PyTorch, diffusers, transformers, accelerate, bitsandbytes

See `quantization/requirements.txt` and `Image-Deepfake-Detectors-Public-Library/environment.yml` for dependencies.
