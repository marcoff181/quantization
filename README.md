# CrispyMcMarkInc — AI-Generated Image Detection Under Quantization

This repository studies how **model quantization** (fp16 / fp8 / fp4) of diffusion models affects **deepfake image detection**. It has two parallel tracks:

1. **Text-to-Image (txt2img)**: Generate fake images from text prompts using quantized models, then run multiple detectors.
2. **Image-to-Image (img2img)**: Generate subtle perturbations of real images using quantized models at low strength (0.3), then fine-tune **CLIP-D** to detect them.

The detection library is a git submodule: [Image-Deepfake-Detectors-Public-Library](https://github.com/grip-unina/ClipBased-SyntheticImageDetection) (GRIP-UNINA).

---

## Project Structure

```
.
├── pipelines/                    # 🟢 Active entry points grouped by task
│   ├── txt2img/                  # Pipeline 1: text-to-image
│   │   ├── generate.py           # Batch txt2img generation via quantized models
│   │   ├── detect.py             # Run CLIP-D + R50_TF + NPR on txt2img outputs
│   │   └── analyze.py            # Analyze txt2img detection results
│   └── img2img/                  # Pipeline 2: image-to-image (main track)
│       ├── prepare_data.py       # Center-crop real images to 1024×1024
│       ├── prepare_splits.py     # Create train/val/test split lists
│       ├── generate.py           # Generate img2img fakes from split lists
│       ├── build_manifests.py    # Build CSV manifests (path,label) for training
│       ├── evaluate.py           # Run CLIP-D detection on img2img outputs
│       ├── evaluate_real.py      # Run all CLIP-D modes on real images (baseline)
│       ├── analyze_default.py    # Analyze default-mode detection results
│       ├── analyze_patches.py    # Analyze patch/grid detection results
│       ├── analyze_modes.py      # Compare all detection modes (ROC/PR curves)
│       ├── analyze_reals.py      # Real-image baseline analysis
│       └── README.md             # Full fine-tuning guide (from README_img2img_finetune.md)
├── generators/                   # Engine code (imported, not run directly)
│   ├── txt2img/
│   │   ├── generate.py           # txt2img generator (was quantization/txt2img.py)
│   │   └── prompts_filtered.txt  # 5771 text prompts
│   ├── img2img/
│   │   ├── generate.py           # img2img generator (was original_img_to_img/original_img2img.py)
│   │   └── img_to_img_prompts.csv
│   ├── inpainting/
│   │   ├── inpaint.py            # Inpainting generator
│   │   ├── generate_masks.py     # CLIPSeg mask generation
│   │   └── new_masks_auto.csv
│   └── shared/
│       ├── utils.py              # Shared model/config constants (was shared_utils.py)
│       └── monitor.py            # GPU/CPU/RAM resource monitor
├── detection/                    # Detector integrations
│   └── submodule/                # Image-Deepfake-Detectors-Public-Library (git submodule)
├── data/                         # Input data (gitignored)
│   ├── real/                     # Cropped real images (1024×1024)
│   │   └── raw/                  # Original uncropped real images
│   ├── fake/
│   │   ├── txt2img/              # txt2img generated images
│   │   └── img2img/              # img2img generated images (train/val/test)
│   ├── splits/                   # Train/val/test text split files
│   └── manifests/                # CSV manifests for CLIP-D training
├── results/                      # Detection result CSVs and metrics
│   ├── txt2img/
│   └── img2img/
├── figures/                      # Plots and analysis figures
│   ├── img2img_default/
│   ├── mode_comparison/
│   └── inpainting/
├── legacy/                       # 🔴 Stale/experimental code (kept for reference)
│   ├── stale_scripts/
│   └── experiments/
├── _paths.py                     # Path definitions (import by scripts)
└── README.md                     # This file
```

---

## Pipeline 1: Quantized txt2img Generation + Detection

### 1.1 Generate images

```bash
python pipelines/txt2img/generate.py \
  --models sd15 sd3 sd35 sdxl \
  --quantizations fp16 fp8 fp4 \
  --prompts 100
```

Calls `generators/txt2img/generate.py` as subprocess for every model×quantization combination. Supported models: `sd15`, `sd3`, `sd35`, `sdxl`, `flux`, `z-image`, `pg25`. Output goes to `data/fake/txt2img/`.

### 1.2 Run detection

```bash
python pipelines/txt2img/detect.py \
  --detectors CLIP-D R50_TF NPR
```

Writes per-image predictions to `results/txt2img/detector_final_results.csv`.

### 1.3 Analyze

```bash
python pipelines/txt2img/analyze.py
```

---

## Pipeline 2: Quantized img2img Generation + CLIP-D Fine-Tuning

This is the main experimental track. Real images from FORLAB are cropped to 1024×1024, split 70/15/15, then img2img fakes are generated at low strength (0.3) using quantized models. CLIP-D is fine-tuned to detect the subtle artifacts.

### 2.1 Dataset preparation

```bash
# Crop real images to 1024×1024
python pipelines/img2img/prepare_data.py

# Split into train/val/test (70/15/15) at the image level
python pipelines/img2img/prepare_splits.py \
  --limit 500 \
  --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15
```

### 2.2 Generate img2img fakes

```bash
python pipelines/img2img/generate.py \
  --splits train val test \
  --models sd15 sd3 sd35 \
  --quantizations fp16 \
  --strength 0.3 --steps 30 --guidance 3.5
```

This calls `generators/img2img/generate.py` for each image. Each real image is perturbed at strength 0.3 — subtle enough that the fake is visually nearly identical to the original.

### 2.3 Build training manifests

```bash
python pipelines/img2img/build_manifests.py
```

Outputs CSVs with columns `[path, label]` (0=real, 1=fake) to `data/manifests/`.

### 2.4 Fine-tune CLIP-D

```bash
python detection/submodule/detectors/CLIP-D/train.py \
  --name img2img_ft_fp16_v2 \
  --arch opencliplinearnextft_clipL14commonpool \
  --task train \
  --device cuda:0 \
  --csv_train data/manifests/img2img_train.csv \
  --csv_val data/manifests/img2img_val.csv \
  --csv_test data/manifests/img2img_test.csv \
  --pretrained_weights detection/submodule/detectors/CLIP-D/checkpoint/pretrained/weights/best.pt \
  --batch_size 8 \
  --lr 5e-5 \
  --backbone_lr 1e-6 \
  --num_epoches 20 \
  --earlystop_epoch 3
```

Key details:
- **Architecture**: `OpenClipLinearFT` — fine-tunable OpenCLIP ViT-L/14 (CommonPool pretrained). Gradients flow through the backbone.
- **Two learning rates**: linear head at 5e-5, backbone at 1e-6 (10× smaller).
- **Early stopping**: patience of 3 epochs; when validation balanced accuracy plateaus, LR is divided by 10 (min 1e-6).
- **Augmentation**: RandomResizedCrop (20% prob) + Random JPEG compression (50% prob), then resize to 224×224 + CLIP normalization.
- **Loss**: BCEWithLogitsLoss. **Optimizer**: Adam (β₁=0.9, weight decay=0.0).

### 2.5 Evaluate

```bash
python detection/submodule/detectors/CLIP-D/test.py \
  --name img2img_ft_fp16_v2 \
  --arch opencliplinearnextft_clipL14commonpool \
  --task test \
  --device cuda:0 \
  --csv_test data/manifests/img2img_test.csv
```

### 2.6 Results (fine-tuned model on held-out test set)

| Metric | Value |
|---|---|
| TPR | 0.991 |
| TNR | 0.960 |
| Accuracy | 0.983 |
| AUC | 0.997 |

### 2.7 Why full-image fine-tuning, not patch/grid

The frozen-baseline CLIP-D (linear head only) plateaus at AUC ~0.65 on this task. Fine-tuning the backbone with a small learning rate lets the feature space adapt to the specific artifacts introduced by the quantized img2img pipeline.

Full-image analysis outperforms patch/grid inference because:
- CLIP-D is designed for **full images** resized to 224×224, not isolated crops
- Crops break global context, causing false positives on real images
- Aggregation (max/majority) amplifies outlier patch errors
- Img2img artifacts are visible in **global statistics** rather than local textures

---

## Pipeline 3: Inpainting (experimental)

```bash
# Generate CLIPSeg masks
python generators/inpainting/generate_masks.py \
  --input_dir /tmp/demo_img/ \
  --output_dir out/inpaint/masks \
  --mask_prompts_file generators/inpainting/new_masks_auto.csv

# Run inpainting
python generators/inpainting/inpaint.py \
  --input_dir /tmp/demo_img/ \
  --mask_dir out/inpaint/masks \
  --prompts_file generators/inpainting/new_masks_auto.csv \
  --models sd15 sd3 \
  --quantization fp16 fp8 fp4 \
  --strength 0.75 --guidance 8.0 --steps 30
```

---

## Detectors (submodule)

All detectors live in `detection/submodule/detectors/`. The project primarily uses:

| Detector | Architecture | Weights |
|---|---|---|
| **CLIP-D** | OpenCLIP ViT-L/14 + linear head | `opencliplinearnext_clipL14commonpool` (frozen) or `opencliplinearnextft_clipL14commonpool` (fine-tuned) |
| **R50_TF** | ResNet-50 with three-filter | `nodown` arch |
| **NPR** | Noise Pattern Residual | — |
| **R50_nodown** | ResNet-50 without stride-2 downsampling | `res50nodown` arch |
| **P2G** | Pixel-to-Gram | — |

---

## Stale / Legacy Code

The following scripts exist in `legacy/` but are **not part of the active pipeline**:

| File | Why it is stale |
|---|---|
| `legacy/stale_scripts/move_images.py` | One-time FORLAB copy; hardcoded NAS path |
| `legacy/stale_scripts/generate_images_img2img.py` | Early flat-folder batch runner; superseded by split-based version |
| `legacy/stale_scripts/img2img.py` | Duplicate img2img implementation (active one is in `generators/img2img/`) |
| `legacy/stale_scripts/components.py` | Debug script for listing model quantizable components |
| `legacy/stale_scripts/original_img2img_run.sh` | One-off shell example |
| `legacy/stale_scripts/manifest_generation_README.md` | Superseded by `pipelines/img2img/README.md` |
| `legacy/stale_scripts/README_1.md` | Superseded by `generators/README.md` |
| `legacy/experiments/quant_comparative/` | Parallel experiment on different paths |

---

## Requirements

Two conda/virtual environments are expected:

- **`detector`** (for pipelines/txt2img/detect.py, pipelines/img2img/evaluate.py, training): PyTorch 2.4, open-clip-torch, scikit-learn, pandas, matplotlib, seaborn
- **`quantization`** (for pipelines/txt2img/generate.py, pipelines/img2img/generate.py, generators/*): PyTorch, diffusers, transformers, accelerate, bitsandbytes

See `generators/requirements.txt` and `detection/submodule/environment.yml` for pinned dependencies.
