from diffusers import DiffusionPipeline

from shared_utils import MODELS

""" Simple script to check what is quantizable from the selected models """

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