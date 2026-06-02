# Img2img manifest generation

This helper builds train/val/test CSV manifests from the split lists and the generated fakes.

## Quick start

```bash
python /media/CrispyMcMarkInc/generate_img2img_manifests.py \
  --split_dir /media/CrispyMcMarkInc/splits \
  --output_dir /media/CrispyMcMarkInc/manifests \
  --fake_root /media/CrispyMcMarkInc/fake-images-img2img
```

## Dry run (counts only)

```bash
python /media/CrispyMcMarkInc/generate_img2img_manifests.py --dry_run
```

## Output

Manifests are written to:
- /media/CrispyMcMarkInc/manifests/img2img_train.csv
- /media/CrispyMcMarkInc/manifests/img2img_val.csv
- /media/CrispyMcMarkInc/manifests/img2img_test.csv
