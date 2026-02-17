from diffusers import DiffusionPipeline

# --- Configuration & Constants ---
MODELS = {
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    # UPDATED: Replaced sd2 with sd3
    "sd3": "stabilityai/stable-diffusion-3-medium-diffusers",
    "sd35": "stabilityai/stable-diffusion-3.5-medium",
    "flux": "black-forest-labs/FLUX.1-dev",
    "sd15": "runwayml/stable-diffusion-v1-5",
}

ignore = [
    "_class_name",
    "_diffusers_version",
    "scheduler",
    "tokenizer",
    "tokenizer_2",
    "tokenizer_3",
    "vae",
    "safety_checker",
    "image_encoder",
]

for model_key, model_name in MODELS.items():
    config = DiffusionPipeline.load_config(model_name)
    components = [comp for comp in config.keys() if comp not in ignore]   
    print(f"Components for {model_key} ({model_name}): {components}")