# Img2Img Comparative Pipeline

The pipeline runs in three stages:

1. Generate `img2img` images across multiple model, quantization, and parameter combinations.
2. Run deepfake detectors on the generated images.
3. Summarize the results and produce tables and plots.


The default paths used by the pipeline are:

- real input images: `/media/SSD_4TB/crispy_storage/CRISPY_DATASET/PreSocial/Real/`
- generated output images: `/media/SSD_4TB/crispy_storage/comparative_images_img2img/`
- detection results: `/media/CrispyMcMarkInc/detector_results.csv`
- figures and summaries: `/media/CrispyMcMarkInc/figures/img2img/`

## 1. Generate images

Run:

```bash
python generate_images.py
```

Useful options:

- `--steps 30 50`
- `--guidance 3.5 5.0`
- `--strength 0.3 0.5`
- `--max_images 2` per prompt
- `--complete` to skip batches that are already complete

This script generates combinations for `sd15`, `sd3`, and `sd35`, using `fp16`, `fp8`, and `fp4` quantization.

## 2. Check missing files

```bash
python check_missing_images.py \
  --steps 30 \
  --guidance 3.5 \
  --strength 0.3 \
  --max_images 2
```
## 3. Run detection

```bash
python run_detection.py --complete
```

Without `--complete`, the results CSV is reinitialized. With `--complete`, the script tries to resume from existing CSV.

## 4. Analyze results

```bash
python analyze_detector_performance.py /path/to/your_file.csv
```

## Recommended workflow

1. `generate_images.py`
2. `check_missing_images.py` if you want to check coverage
3. `run_detection.py --complete`
4. `analyze_detector_performance.py ./detector_results.csv`

## Practical notes

- The main parameters are hardcoded at the top of the scripts, so if you change paths or environments it is best to update them there.
- Generated file names must match the pattern expected by `run_detection.py`; otherwise the metadata in the results will be marked as `unknown`.