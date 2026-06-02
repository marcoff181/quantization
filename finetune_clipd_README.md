# Fine-tune CLIP-D on img2img data

This guide fine-tunes CLIP-D using CSV manifests generated from your img2img splits.

## Inputs

- Pretrained checkpoint (best.pt) from the detector plugin.
- CSV manifests:
  - /media/CrispyMcMarkInc/manifests/img2img_train.csv
  - /media/CrispyMcMarkInc/manifests/img2img_val.csv
  - /media/CrispyMcMarkInc/manifests/img2img_test.csv

## Command (train)

```bash
cd /media/CrispyMcMarkInc/Image-Deepfake-Detectors-Public-Library/detectors/CLIP-D

python train.py \
  --name clipd_img2img_ft \
  --task train \
  --device cuda:0 \
  --arch opencliplinearnext_clipL14commonpresool \
  --csv_train /media/CrispyMcMarkInc/manifests/img2img_train.csv \
  --csv_val /media/CrispyMcMarkInc/manifests/img2img_val.csv \
  --csv_test /media/CrispyMcMarkInc/manifests/img2img_test.csv \
  --pretrained_weights /path/to/best.pt \
  --batch_size 32 \
  --num_threads 8 \
  --lr 1e-5 \
  --weight_decay 0.0 \
  --num_epoches 12 \
  --earlystop_epoch 4 \
  --resizeSize 224 \
  --resize_prob 0.0 \
  --cmp_prob 0.0
```

Notes:
- Pick a new `--name` so you do not overwrite the original best.pt.
- The training outputs go to `checkpoint/<name>/weights`.

## Evaluation

The training script runs validation every epoch and saves `best.pt` in the new checkpoint folder.
You can later use that new `best.pt` for evaluation against your test split.
