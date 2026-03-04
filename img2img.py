import argparse
import os
import glob
import random
import torch
import gc
from PIL import Image

from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
)

from diffusers.quantizers import PipelineQuantizationConfig
from resource_monitor import ResourceMonitor

# --- Configuration & Constants ---
MODELS = {
    'sd15': 'runwayml/stable-diffusion-v1-5',
    'sd3': 'stabilityai/stable-diffusion-3-medium-diffusers',
    'sd35': 'stabilityai/stable-diffusion-3.5-medium',
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
    def __init__(self, model_key, quantization, device=None, monitor=None):
        self.model_key = model_key
        self.quantization = quantization
        self.monitor = monitor
        
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
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
                self.model_key,
                self.quantization,
                "load_model",
                extra={"model_size_mb": self.model_size_mb},
            )

    def _get_dtype(self):
        if self.device == "cpu":
            return torch.float32

        if self.model_key in ["sd3", "sd35"]:
            if self.device == "cuda" and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16

        return torch.float16

    def load_pipeline(self):
        print(f"Loading {self.model_key} with {self.quantization} quantization...")
        
        model_id = MODELS[self.model_key]
        q_config = quantization_levels(self.model_key)[self.quantization]
        dtype = self._get_dtype()
        
        try:
            if self.model_key in ["sd3", "sd35"]:
                self.pipeline_t2i = AutoPipelineForText2Image.from_pretrained(
                    model_id,
                    quantization_config=q_config,
                    torch_dtype=dtype,
                )
            else:
                variant = (
                    "fp16"
                    if (self.quantization == "fp16" and self.device != "cpu")
                    else None
                )
                self.pipeline_t2i = AutoPipelineForText2Image.from_pretrained(
                    model_id,
                    quantization_config=q_config,
                    torch_dtype=dtype,
                    variant=variant,
                    use_safetensors=True,
                )

            self.model_size_mb = self._compute_model_size(self.pipeline_t2i)
            print(f"  Model parameter size: {self.model_size_mb:.1f} MB")

            if self.quantization == "fp8":
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
        total_bytes = 0
        seen = set()
        for _, component in pipeline.components.items():
            if not hasattr(component, "parameters"):
                continue
            for p in component.parameters():
                p_id = p.data_ptr()
                if p_id in seen:
                    continue
                seen.add(p_id)
                total_bytes += p.nelement() * p.element_size()
        return total_bytes / 1024**2

    def generate(self, prompt, image, strength=0.3, guidance_scale=3.5, steps=30, seed=None):
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator("cpu").manual_seed(seed)

        print(f"Generating with seed: {seed}")

        if self.monitor:
            self.monitor.start_timer()

        kwargs = {
            "prompt": prompt,
            "image": image,
            "strength": strength,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator 
        }

        result = self.pipeline_i2i(**kwargs).images[0]
        if self.monitor:
            self.monitor.record(
                self.model_key,
                self.quantization,
                "generate",
                extra={"seed": seed, "steps": steps, "model_size_mb": self.model_size_mb},
            )
        return result

def main():
    parser = argparse.ArgumentParser(description="Generate images using various Stable Diffusion models (Image-to-Image only).")
    parser.add_argument("--prompt", type=str, default="", help="Prompt for generation.")
    parser.add_argument("--models", nargs='+', default=['sd15'], choices=MODELS.keys(), help="Models to use.")
    parser.add_argument("--quantization", nargs='+', default=['fp16'], choices=quantization_levels("sd35").keys(), help="Quantization levels.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images.")
    parser.add_argument("--output_dir", type=str, default="output_img2img", help="Directory for output images.")
    parser.add_argument("--strength", type=float, default=0.3, help="Denoising strength (0.0 to 1.0).")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps.")
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance scale.")
    parser.add_argument("--device", type=str, default=None, help="Device to use.")
    parser.add_argument("--seed", type=int, default=123, help="Fixed seed for reproducibility.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Input directory {args.input_dir} does not exist.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device:
        monitor_device = args.device
    elif torch.cuda.is_available():
        monitor_device = "cuda"
    elif torch.backends.mps.is_available():
        monitor_device = "mps"
    else:
        monitor_device = "cpu"

    monitor = ResourceMonitor(device=monitor_device)
    
    input_images = sorted(glob.glob(os.path.join(args.input_dir, '*.jpg')) + 
                          glob.glob(os.path.join(args.input_dir, '*.png')) +
                          glob.glob(os.path.join(args.input_dir, '*.PNG')) +
                          glob.glob(os.path.join(args.input_dir, '*.JPG')) +
                          glob.glob(os.path.join(args.input_dir, '*.jpeg')))

    if not input_images:
        print(f"No images found in {args.input_dir}")
        return

    for model_key in args.models:
        for quant in args.quantization:
            seed = args.seed
            try:
                generator = ImageGenerator(model_key, quant, device=args.device, monitor=monitor)
                
                print(f"Processing images with {model_key} ({quant})...")
                
                for i, img_path in enumerate(input_images):
                    try:
                        print(f"  Processing {os.path.basename(img_path)}...")
                        
                        input_img = Image.open(img_path).convert("RGB")
                        width, height = input_img.size

                        # Center crop 1024x1024
                        if width > 1024 and height > 1024:
                            left = (width - 1024)/2
                            top = (height - 1024)/2
                            right = (width + 1024)/2
                            bottom = (height + 1024)/2
                            input_img = input_img.crop((left, top, right, bottom))
                        
                        # current_seed = args.seed + i if args.seed is not None else None
                        
                        output_img = generator.generate(
                            args.prompt, 
                            image=input_img,
                            strength=args.strength,
                            steps=args.steps, 
                            guidance_scale=args.guidance, 
                            seed=seed
                        )
                        
                        seed += 1
                        
                        base_name = os.path.splitext(os.path.basename(img_path))[0]
                        seed_str = f"_seed{seed}" if seed is not None else ""
                        out_name = f"{args.output_dir}/{base_name}_{model_key}_{quant}{seed_str}.png"
                        output_img.save(out_name)
                        print(f"    Saved {out_name}")
                        
                    except Exception as e_img:
                        print(f"    Failed to process {os.path.basename(img_path)}: {e_img}")

                flush()
                del generator
                torch.cuda.empty_cache()
                flush()
                
            except Exception as e:
                print(f"Failed to load/run {model_key} with {quant}: {e}")
                flush()

    metrics_path = os.path.join(args.output_dir, "resource_metrics.csv")
    monitor.save_csv(metrics_path)
    monitor.summary()

if __name__ == "__main__":
    main()
