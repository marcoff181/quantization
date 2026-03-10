import argparse
import os
import random
import torch
import gc

from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    Flux2KleinPipeline,
    FluxPipeline,
    ZImagePipeline,
)

from diffusers.quantizers import PipelineQuantizationConfig
from resource_monitor import ResourceMonitor

# --- Configuration & Constants ---
MODELS = {
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    # UPDATED: Replaced sd2 with sd3
    "sd3": "stabilityai/stable-diffusion-3-medium-diffusers",
    "sd35": "stabilityai/stable-diffusion-3.5-medium",
    "flux": "black-forest-labs/FLUX.1-dev",
    "sd15": "runwayml/stable-diffusion-v1-5",
    "z-image": "Tongyi-MAI/Z-Image",
    "flux2": "black-forest-labs/FLUX.2-klein-9B"
}

# High-quality defaults used when --steps/--guidance are not explicitly provided.
QUALITY_PRESETS = {
    "sd15": {"steps": 50, "guidance": 8.0},
    "sdxl": {"steps": 50, "guidance": 7.5},
    "sd3": {"steps": 45, "guidance": 6.0},
    "sd35": {"steps": 45, "guidance": 6.0},
    "flux": {"steps": 50, "guidance": 4.0},
    "z-image": {"steps": 50, "guidance": 4.0},
    "flux2": {"steps": 4, "guidance": 1.0}
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
    elif model_key in  ["z-image", "flux2"]:
        return ["text_encoder", "transformer"]
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


def get_quality_params(model_key, user_steps, user_guidance):
    preset = QUALITY_PRESETS.get(model_key, {"steps": 50, "guidance": 7.0})
    steps = user_steps if user_steps is not None else preset["steps"]
    guidance = user_guidance if user_guidance is not None else preset["guidance"]
    return steps, guidance


def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


class ImageGenerator:
    def __init__(self, model_key, quantization, device=None, monitor=None):
        self.model_key = model_key
        self.quantization = quantization
        self.monitor = monitor

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
        self.model_size_mb = 0.0

        if self.monitor:
            self.monitor.start_timer()
            
        self.load_pipeline()
        if self.monitor:
            self.monitor.record(
                self.model_key, self.quantization, "load_model",
                extra={"model_size_mb": self.model_size_mb}
            )

    def _get_dtype(self):
        # CPU
        if self.device == "cpu":
            return torch.float32

        # Flux and SD3/SD3.5 support bfloat16 on compatible hardware, otherwise fallback to float16
        if self.model_key in ["flux", "sd3", "sd35", "z-image", "flux2"]:
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
                # if self.model_key == "sd35" and self.quantization != "fp16":
                #     target_model = "stabilityai/stable-diffusion-3.5-large"

                self.pipeline_t2i = AutoPipelineForText2Image.from_pretrained(
                    target_model,
                    quantization_config=q_config,
                    torch_dtype=dtype,
                )
            
            elif self.model_key == "z-image":
                self.pipeline_t2i = ZImagePipeline.from_pretrained(
                    model_id,
                    quantization_config=q_config,
                    torch_dtype=dtype,
                )

            elif self.model_key == "flux2":
                self.pipeline_t2i = Flux2KleinPipeline.from_pretrained(
                    model_id,
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
                )  
                
            # Compute actual model parameter sizes BEFORE offload
            self.model_size_mb = self._compute_model_size(self.pipeline_t2i)
            print(f"  Model parameter size: {self.model_size_mb:.1f} MB")

            # offload strategy: bnb 8bit modules cannot be moved to CPU
            if self.quantization == "fp8":
                # For 8bit quantized models, keep on device (no offload possible)
                try:
                    self.pipeline_t2i.to(self.device)
                    print(f"  Moved fp8 pipeline to {self.device}")
                except Exception:
                    print(f"  Warning: could not move fp8 pipeline to {self.device}, already on device")
            else:
                try:
                    self.pipeline_t2i.enable_model_cpu_offload()
                    print(f"  Enabled CPU offload for {self.model_key}")
                except Exception:
                    print("  Warning: cpu offload failed, trying sequential offload...")
                    try:
                        self.pipeline_t2i.enable_sequential_cpu_offload()
                        print(f"  Enabled sequential CPU offload for {self.model_key}")
                    except Exception as e2:
                        print(f"  Warning: sequential offload also failed: {e2}")
                        self.pipeline_t2i.to(self.device)

            # Convert to Image2Image pipeline
            self.pipeline_i2i = AutoPipelineForImage2Image.from_pipe(self.pipeline_t2i)

        except Exception as e:
            print(f"Error loading pipeline: {e}")
            raise e

    @staticmethod
    def _compute_model_size(pipeline):
        """Compute total size (MB) of all model parameters in the pipeline,
        accounting for quantized storage (e.g. int8, uint8 for bnb)."""
        total_bytes = 0
        seen = set()
        for name, component in pipeline.components.items():
            if not hasattr(component, 'parameters'):
                continue
            for p in component.parameters():
                p_id = p.data_ptr()
                if p_id in seen:
                    continue
                seen.add(p_id)
                # element_size() returns actual storage bytes per element
                # (1 for int8/uint8 from bitsandbytes, 2 for fp16, 4 for fp32)
                total_bytes += p.nelement() * p.element_size()
        return total_bytes / 1024**2

    def generate(
        self, prompt, image=None, strength=0.3, guidance_scale=3.5, steps=30, seed=None
    ):

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator("cpu").manual_seed(seed)

        print(f"Generating with seed: {seed}")

        if self.monitor:
            self.monitor.start_timer()

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
                result = self.pipeline_i2i(**kwargs).images[0]
            else:
                result = self.pipeline_t2i(**kwargs).images[0]

            if self.monitor:
                self.monitor.record(
                    self.model_key, self.quantization, "generate",
                    extra={"seed": seed, "steps": steps, "model_size_mb": self.model_size_mb}
                )
            return result
        except Exception as e:
            print(f"Text-to-Image not supported for this model configuration: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate images using multiple diffusion models.")
    parser.add_argument("--models",nargs="+",default=MODELS.keys(),choices=MODELS.keys(),help="Models to use. Default is all available models")
    parser.add_argument("--prompt", type=str, default="",help="Prompt for generation.")
    parser.add_argument("--prompts",type=int, default=100,help="If --prompt is not provided, how many prompts to load automatically")
    parser.add_argument("--prompts_file",type=str, default="prompts_filtered.txt",help="File to load prompts from if --prompt is not provided")
    parser.add_argument("--quantization",nargs="+", default=quantization_levels("sd35").keys(),choices=quantization_levels("sd35").keys(),help="Quantization levels.")
    parser.add_argument("--output_dir",type=str, default="/media/SSD_4TB/crispy_storage",help="Directory for output images.")
    parser.add_argument("--strength", type=float, default=0.3, help="Strength for img2img only (not used in txt2img mode).")
    parser.add_argument("--steps", type=int, default=None, help="Inference steps. If omitted, uses high-quality model defaults.")
    parser.add_argument("--guidance", type=float, default=None, help="Guidance scale. If omitted, uses high-quality model defaults.")
    parser.add_argument("--device", type=str, default=None, help="Device to use.")
    parser.add_argument("--seed", type=int, default=123, help="Fixed seed for reproducibility.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine device once for the monitor
    if args.device:
        monitor_device = args.device
    elif torch.cuda.is_available():
        monitor_device = "cuda"
    elif torch.backends.mps.is_available():
        monitor_device = "mps"
    else:
        monitor_device = "cpu"

    monitor = ResourceMonitor(device=monitor_device)

    for model_key in args.models:
        for quant in args.quantization:
            seed = args.seed  # Reset seed for each model to ensure comparability
            try:
                generator = ImageGenerator(model_key, quant, device=args.device, monitor=monitor)

                effective_steps, effective_guidance = get_quality_params(
                    model_key,
                    args.steps,
                    args.guidance,
                )

                print(f"Generating txt2img with {model_key} ({quant})...")
                print(f"Quality params -> steps: {effective_steps}, guidance: {effective_guidance}")

                # TODO: add check to avoid that args.prompts is larger than the number of lines in prompts_filtered.txt and to avoid loading too many prompts into memory at once if the file is huge (e.g. load in batches)
                if args.prompt == "" and args.prompts > 0:
                    print(f"Loading prompts from {args.prompts_file}...")
                    with open(args.prompts_file, "r") as file:
                        # load only if line is not empty and strip whitespace from each line
                        prompts = [line.strip() for line in file if line.strip()]
                        
                    if len(prompts) < args.prompts:
                        print(f"Warning: requested {args.prompts} prompts but only found {len(prompts)} in file. Using all available prompts.")
                else:
                    prompts = [args.prompt]


                print(f"Generating {len(prompts)} images using {model_key} ({quant})...")
                
                for i, prompt in enumerate(prompts):
                    img = generator.generate(
                        prompt,
                        steps=effective_steps,
                        guidance_scale=effective_guidance,
                        seed=seed,
                    )
                    
                    # if somebody can explain it I will put it back
                    seed += 1  # ensure different seed for each model/quantization combo, but keep it deterministic across runs
                    
                    if img is None:
                        print(f"Generation failed for model {model_key} with quantization {quant} on prompt {i+1}/{len(prompts)}: '{prompt}'")
                        continue

                    out_name = f"{args.output_dir}/{model_key}_{quant}_p{i}_seed{seed-1}.png"
                    
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

    # Save metrics and print summary
    metrics_path = os.path.join(args.output_dir, "resource_metrics.csv")
    monitor.save_csv(metrics_path)
    monitor.summary()


if __name__ == "__main__":
    main()
