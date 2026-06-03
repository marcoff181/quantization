from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Generators
GENERATORS = ROOT / "generators"
TXT2IMG_GEN = GENERATORS / "txt2img"
IMG2IMG_GEN = GENERATORS / "img2img"
INPAINT_GEN = GENERATORS / "inpainting"
SHARED_GEN = GENERATORS / "shared"

# Data
DATA = ROOT / "data"
REAL = DATA / "real"
FAKE_TXT2IMG = DATA / "fake" / "txt2img"
FAKE_IMG2IMG = DATA / "fake" / "img2img"
SPLITS = DATA / "splits"
MANIFESTS = DATA / "manifolds"

# Detection
DETECTION = ROOT / "detection"
DETECTOR_SUBMODULE = DETECTION / "submodule"
DETECTOR_CONFIGS = DETECTION / "configs"
DETECTOR_WEIGHTS = DETECTION / "weights"
PRETRAINED_WEIGHTS = DETECTOR_WEIGHTS / "pretrained"
FT_V1_WEIGHTS = DETECTOR_WEIGHTS / "img2img_ft_v1"
FT_V2_WEIGHTS = DETECTOR_WEIGHTS / "img2img_ft_v2"

# Pipelines
PIPELINES = ROOT / "pipelines"
TXT2IMG_PIPELINE = PIPELINES / "txt2img"
IMG2IMG_PIPELINE = PIPELINES / "img2img"

# Results
RESULTS = ROOT / "results"
TXT2IMG_RESULTS = RESULTS / "txt2img"
IMG2IMG_RESULTS = RESULTS / "img2img"
COMPARISON_RESULTS = RESULTS / "comparison"

# Figures
FIGURES = ROOT / "figures"
TXT2IMG_FIGURES = FIGURES / "txt2img"
IMG2IMG_DEFAULT_FIGURES = FIGURES / "img2img_default"
IMG2IMG_PATCH_FIGURES = FIGURES / "img2img_patches"
MODE_COMPARISON_FIGURES = FIGURES / "mode_comparison"

# Legacy
LEGACY = ROOT / "legacy"
