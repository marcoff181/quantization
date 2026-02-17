import argparse
import os
import random
import torch
import gc

from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    FluxPipeline,
)

from diffusers.quantizers import PipelineQuantizationConfig

# --- Configuration & Constants ---
MODELS = {
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    # UPDATED: Replaced sd2 with sd3
    "sd3": "stabilityai/stable-diffusion-3-medium-diffusers",
    "sd35": "stabilityai/stable-diffusion-3.5-medium",
    "flux": "black-forest-labs/FLUX.1-dev",
    "sd15": "runwayml/stable-diffusion-v1-5",
}


def get_components(model_key):
    if model_key == "sdxl":
        return ["text_encoder", "text_encoder_2", "unet"]
    elif model_key == "sd15":
        return ["text_encoder", "unet"]
    elif model_key == "flux":
        return ["text_encoder", "text_encoder_2", "transformer"]
    elif model_key in ["sd3", "sd35"]:
        return ["text_encoder", "text_encoder_2", "text_encoder_3", "transformer"]
    else:
        raise ValueError(f"Unknown model key: {model_key}")


def quantization_levels(model_key):
    return {
        "fp16": None,
        "fp8": PipelineQuantizationConfig(
            quant_backend="bitsandbytes_8bit",
            quant_kwargs={
                "load_in_8bit": True,
            },
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


def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


class ImageGenerator:
    def __init__(self, model_key, quantization, device=None):
        self.model_key = model_key
        self.quantization = quantization

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        self.pipeline_i2i = None
        self.pipeline_t2i = None

        self.load_pipeline()

    def _get_dtype(self):
        # CPU
        if self.device == "cpu":
            return torch.float32

        # Flux and SD3/SD3.5 support bfloat16 on compatible hardware, otherwise fallback to float16
        if self.model_key in ["flux", "sd3", "sd35"]:
            if self.device == "cuda" and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16

        # SDXL and SD1.5 support float16 on GPU, but we keep float32 for CPU
        return torch.float16

    def load_pipeline(self):
        print(f"Loading {self.model_key} with {self.quantization} quantization...")

        model_id = MODELS[self.model_key]
        q_config = quantization_levels(self.model_key)[self.quantization]
        dtype = self._get_dtype()

        try:
            # Case 1: flux
            if self.model_key == "flux":
                self.pipeline_t2i = FluxPipeline.from_pretrained(
                    model_id,
                    quantization_config=q_config,
                    torch_dtype=dtype,
                )

            # Case 2: SD3 and SD3.5
            elif self.model_key in ["sd35", "sd3"]:
                target_model = model_id
                if self.model_key == "sd35" and self.quantization != "fp16":
                    target_model = "stabilityai/stable-diffusion-3.5-large"

                self.pipeline_t2i = AutoPipelineForText2Image.from_pretrained(
                    target_model,
                    quantization_config=q_config,
                    torch_dtype=dtype,
                )

            # Case 3: SDXL and SD1.5
            else:
                variant = (
                    "fp16"
                    if (self.quantization == "fp16" and self.device != "cpu")
                    else None
                )

                self.pipeline_t2i = AutoPipelineForText2Image.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    variant=variant,
                    use_safetensors=True,
                )  # .to(self.device) # if the model is really big, the vram is saturated before offload can kick in

            # offload for all the models
            self.pipeline_t2i.enable_model_cpu_offload()

            # Convert to Image2Image pipeline
            self.pipeline_i2i = AutoPipelineForImage2Image.from_pipe(self.pipeline_t2i)

        except Exception as e:
            print(f"Error loading pipeline: {e}")
            raise e

    def generate(
        self, prompt, image=None, strength=0.3, guidance_scale=3.5, steps=30, seed=None
    ):

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator("cpu").manual_seed(seed)

        print(f"Generating with seed: {seed}")

        kwargs = {
            "prompt": prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }

        try:
            if image is not None:
                kwargs["image"] = image
                kwargs["strength"] = strength
                return self.pipeline_i2i(**kwargs).images[0]

            else:
                return self.pipeline_t2i(**kwargs).images[0]
        except Exception as e:
            print(f"Text-to-Image not supported for this model configuration: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate images using various Stable Diffusion models.")
    parser.add_argument("--models",nargs="+",default=MODELS.keys(),choices=MODELS.keys(),help="Models to use. Default is all available models")
    parser.add_argument("--prompt", type=str, default="",help="Prompt for generation.")
    parser.add_argument("--prompts",type=int,default=100,help="If --prompt is not provided, how many prompts to load automatically")
    parser.add_argument("--quantization",nargs="+",default=quantization_levels("sd35").keys(),choices=quantization_levels("sd35").keys(),help="Quantization levels.")
    parser.add_argument("--output_dir",type=str,default="Face2Fake_pt2/output",help="Directory for output images.")
    parser.add_argument("--strength", type=float, default=0.3, help="Strength for img2img.")
    parser.add_argument("--steps", type=int, default=60, help="Inference steps.")
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance scale.")
    parser.add_argument("--device", type=str, default=None, help="Device to use.")
    parser.add_argument("--seed", type=int, default=123, help="Fixed seed for reproducibility.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for model_key in args.models:
        for quant in args.quantization:
            try:
                generator = ImageGenerator(model_key, quant, device=args.device)

                print(f"Generating txt2img with {model_key} ({quant})...")

                if args.prompt == "" and args.prompts > 0:
                    with open("prompts_general.txt", "r") as file:
                        prompts = [file.readline().strip() for _ in range(args.prompts)]
                else:
                    prompts = [args.prompt]

                # if somebody can explain it I will put it back
                # current_seed = args.seed + i if args.seed is not None else None

                print(f"Generating {len(prompts)} images using {model_key} ({quant})...")
                
                for i, prompt in enumerate(prompts):
                    img = generator.generate(
                        prompt,
                        steps=args.steps,
                        guidance_scale=args.guidance,
                        seed=args.seed,
                    )

                    out_name = f"{args.output_dir}/{model_key}_{quant}_p{i}_seed{args.seed}.png"
                    
                    if os.path.exists(out_name):
                        print(f"[{i+1}/{len(prompts)}] File {out_name} already exists, replacing...")
                    else:
                        print(f"[{i+1}/{len(prompts)}] Saved image {out_name}")
                        
                    img.save(out_name)

                del generator
                torch.cuda.empty_cache()
                flush()

            except Exception as e:
                print(f"Failed to run {model_key} with {quant}: {e}")
                flush()


if __name__ == "__main__":
    main()
