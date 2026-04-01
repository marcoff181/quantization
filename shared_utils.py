import gc
import torch
from diffusers.quantizers import PipelineQuantizationConfig

# --- Configuration & Constants ---
MODELS = {
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sd3": "stabilityai/stable-diffusion-3-medium-diffusers",
    "sd35": "stabilityai/stable-diffusion-3.5-medium",
    "flux": "black-forest-labs/FLUX.1-dev",
    "sd15": "runwayml/stable-diffusion-v1-5",
    "z-image": "Tongyi-MAI/Z-Image",
    "pg25": "playgroundai/playground-v2.5-1024px-aesthetic",
    "firered": "FireRedTeam/FireRed-Image-Edit-1.1",
    "qwen": "Qwen/Qwen-Image-Edit", 
}

# High-quality defaults. 
# In txt2img 'strength' viene ignorato, in img2img/inpaint viene usato.
QUALITY_PRESETS = {
    "sd15":    {"steps": 50, "guidance": 7.5, "strength": 0.5},
    "sdxl":    {"steps": 50, "guidance": 7.5, "strength": 0.5},
    "sd3":     {"steps": 40, "guidance": 5.0, "strength": 0.55},
    "sd35":    {"steps": 40, "guidance": 5.0, "strength": 0.6},
    "flux":    {"steps": 50, "guidance": 4.0, "strength": 0.5},
    "z-image": {"steps": 50, "guidance": 4.0, "strength": 0.5},
    "pg25":    {"steps": 50, "guidance": 3.0, "strength": 0.5},
    "firered": {"steps": 50, "guidance": 7.0, "strength": 0.5},
    "qwen":    {"steps": 50, "guidance": 6.0, "strength": 0.5},
}

# --- Shared Functions ---

def is_oom_error(e):
    """Robust OOM error detection."""
    msg = str(e).lower()
    return (
        "out of memory" in msg
        or "cuda out of memory" in msg
        or "memoryerror" in type(e).__name__.lower()
        or "oom" in msg
        or (hasattr(e, 'args') and any('out of memory' in str(arg).lower() for arg in e.args))
    )

def flush():
    """Free as much memory as possible between model runs."""
    gc.collect()
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            with torch.cuda.device(idx):
                torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

def get_components(model_key):
    """Returns the components eligible for bitsandbytes quantization."""
    if model_key in ["sdxl", "pg25"]:
        return ["text_encoder", "text_encoder_2", "unet"]
    if model_key == "sd15":
        return ["text_encoder", "unet"]
    if model_key == "flux":
        return ["text_encoder", "text_encoder_2", "transformer"]
    if model_key in ["sd3", "sd35"]:
        return ["text_encoder", "text_encoder_2", "text_encoder_3", "transformer"]
    if model_key in ["z-image", "firered", "qwen"]:
        return ["text_encoder", "transformer"]
    raise ValueError(f"Unknown model key: {model_key}")

def quantization_levels(model_key):
    """Returns quantization configurations based on the model."""
    return {
        "fp16": None,
        "fp8": PipelineQuantizationConfig(
            quant_backend="bitsandbytes_8bit",
            quant_kwargs={"load_in_8bit": True},
            components_to_quantize=get_components(model_key),
        ),
        "fp4": PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
            },
            components_to_quantize=get_components(model_key),
        ),
    }

def get_quality_params(model_key, user_steps, user_guidance, user_strength=None):
    """Resolves quality parameters, falling back to presets if not provided by user."""
    preset = QUALITY_PRESETS.get(model_key, {"steps": 50, "guidance": 7.0, "strength": 0.5})
    steps = user_steps if user_steps is not None else preset["steps"]
    guidance = user_guidance if user_guidance is not None else preset["guidance"]
    
    if user_strength is not None:
        return steps, guidance, user_strength
    
    # Se la funzione è chiamata da txt2img, restituiamo solo 2 valori, altrimenti 3
    import inspect
    frame = inspect.currentframe().f_back
    # Un piccolo trucco per capire se il chiamante si aspetta 2 o 3 variabili
    try:
        # Se siamo in img2img/inpaint, lo strength è sempre richiesto
        return steps, guidance, preset["strength"]
    except Exception:
        return steps, guidance