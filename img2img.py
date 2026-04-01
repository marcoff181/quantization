import argparse
import os
import csv
import random
import torch
import inspect
import gc
from PIL import Image, ImageOps

from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    DiffusionPipeline,
    QwenImageEditPipeline  # --- ADDED QWEN ---
)

from inpaint import resize_with_aspect_ratio_padding, restore_output_to_original, mean_abs_pixel_diff

from diffusers.quantizers import PipelineQuantizationConfig
from resource_monitor import ResourceMonitor

from shared_utils import (
    MODELS, 
    QUALITY_PRESETS, 
    is_oom_error, 
    flush, 
    quantization_levels, 
    get_quality_params
)

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

        # --- FIX 1: Add firered and qwen to use bfloat16 to prevent black images ---
        if self.model_key in ["sd3", "sd35", "firered", "qwen"]:
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
            if self.model_key == "firered":
                # --- FIX 5: Added trust_remote_code=True for custom QwenImageEditPlusPipeline ---
                self.pipeline_t2i = DiffusionPipeline.from_pretrained(
                    model_id,
                    quantization_config=q_config,
                    torch_dtype=dtype,
                    use_safetensors=True,
                    trust_remote_code=True 
                )
            elif self.model_key == "qwen":
                # --- ADDED QWEN ---
                self.pipeline_t2i = QwenImageEditPipeline.from_pretrained(
                    model_id,
                    quantization_config=q_config,
                    torch_dtype=dtype,
                )
            elif self.model_key in ["sd3", "sd35"]:
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

            # Disable progress bar config based on snippet if it's Qwen
            if self.model_key == "qwen":
                self.pipeline_t2i.set_progress_bar_config(disable=None)

            # --- FIX 2: Bypass from_pipe for native instruct models ----
            if self.model_key in ["firered", "qwen"]:
                self.pipeline_i2i = self.pipeline_t2i
                print(f"  Preserved native pipeline architecture for {self.model_key}")
            else:
                self.pipeline_i2i = AutoPipelineForImage2Image.from_pipe(self.pipeline_t2i)
                print("  Converted to Image2Image pipeline")
            # -------------------------------------------------------------
            
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
            
        if self.model_key in ["firered", "qwen"]:
            # Instruct models often require more steps to remove noise/artifacts.
            use_steps = steps if steps >= 50 else 50
            if steps < 50: 
                print(f"  Warning: Forcing steps to 50 for {self.model_key} quality.")

            use_guidance = guidance_scale if guidance_scale >= 4.0 else 4.0
            if guidance_scale < 4.0:
                print(f"  Warning: Forcing guidance to 4.0 for {self.model_key} convergence.")

            kwargs = {
                "prompt": prompt,
                "image": image,
                "num_inference_steps": use_steps,  
                "generator": generator 
            }
            
            # Analizziamo dinamicamente quali argomenti accetta la pipeline "custom"
            accepted_params = inspect.signature(self.pipeline_i2i.__call__).parameters
            
            # 1. Gestione Negative Prompt
            if "negative_prompt" in accepted_params:
                kwargs["negative_prompt"] = " " if self.model_key == "qwen" else ""

            # 2. Gestione Guidance Scale (Testo)
            if self.model_key == "qwen" and "true_cfg_scale" in accepted_params:
                kwargs["true_cfg_scale"] = use_guidance
            elif "guidance_scale" in accepted_params:
                kwargs["guidance_scale"] = use_guidance

            # 3. Gestione Fedeltà all'Immagine (Il tuo 'strength')
            if "image_guidance_scale" in accepted_params:
                # Nei modelli Instruct, valori PIÙ ALTI di image_guidance significano PIÙ fedeltà all'originale.
                # Mappiamo il tuo strength (0.0-1.0, dove basso = fedele) in image_guidance (es. 1.0-2.0).
                mapped_img_guidance = 1.0 + (1.0 - strength) 
                kwargs["image_guidance_scale"] = mapped_img_guidance
                print(f"    [Instruct] Mapped user strength {strength} -> image_guidance_scale {mapped_img_guidance:.1f}")
            elif "strength" in accepted_params:
                # Se per caso la pipeline supporta lo strength classico, passiamolo
                kwargs["strength"] = strength

            # Use inference mode for instruct models (as per Qwen snippet)
            with torch.inference_mode():
                result = self.pipeline_i2i(**kwargs).images[0]

        else:
            # For SD15/SD3/SD35, standard parameters remain
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
            actual_steps = kwargs.get("num_inference_steps", steps)
            self.monitor.record(
                self.model_key,
                self.quantization,
                "generate",
                extra={"seed": seed, "steps": actual_steps, "model_size_mb": self.model_size_mb},
            )
            
        return result


def main():
    parser = argparse.ArgumentParser(description="Generate images using various Stable Diffusion models (Image-to-Image only).")
    parser.add_argument("--prompts_csv", type=str, required=True, help="CSV file mapping image filenames to prompts.")
    parser.add_argument("--models", nargs='+', default=['sd15'], choices=MODELS.keys(), help="Models to use.")
    parser.add_argument("--quantization", nargs='+', default=['fp16'], choices=quantization_levels("sd35").keys(), help="Quantization levels.")
    parser.add_argument("--output_dir", type=str, default="output_img2img", help="Directory for output images.")
    parser.add_argument("--strength", type=float, default=None, help="Denoising strength (0.0 to 1.0).")
    parser.add_argument("--steps", type=int, default=None, help="Inference steps.")
    parser.add_argument("--guidance", type=float, default=None, help="Guidance scale.")
    parser.add_argument("--device", type=str, default=None, help="Device to use.")
    parser.add_argument("--seed", type=int, default=123, help="Fixed seed for reproducibility.")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum number of images to process from CSV (default: all).")

    parser.add_argument("--preserve_aspect_ratio", action="store_true", help="Preserve image aspect ratio with smart padding.")
    parser.add_argument("--max_dimension", type=int, default=512, help="Maximum dimension when preserving aspect ratio (default: 512). Longer side won't exceed this.")
    parser.add_argument("--pad_color", type=str, default="128,128,128", help="RGB padding color as 'R,G,B' when preserving aspect ratio (default: 128,128,128 gray).")
    

    args = parser.parse_args()
    
    try:
        pad_colors = args.pad_color.split(',')
        args.pad_color_rgb = tuple(int(c.strip()) for c in pad_colors)
        if len(args.pad_color_rgb) != 3 or any(c < 0 or c > 255 for c in args.pad_color_rgb):
            raise ValueError
    except:
        print(f"Invalid pad_color: {args.pad_color}. Using default 128,128,128")
        args.pad_color_rgb = (128, 128, 128)

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

    # Read CSV mapping image filenames to prompts
    csv_rows = []
    with open(args.prompts_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            csv_rows.append(row)

    if args.max_images:
        csv_rows = csv_rows[:args.max_images]
        print(f"Processing max {args.max_images} image-prompt pairs.")

    base_img_dir = "/media/NAS/TrueFake/PreSocial/Real/FORLAB/"

    for model_key in args.models:
        for quant in args.quantization:
            seed = args.seed
            try:
                generator = ImageGenerator(model_key, quant, device=args.device, monitor=monitor)
                print(f"Processing images with {model_key} ({quant})...")
                for i, row in enumerate(csv_rows):
                    try:
                        img_path = os.path.join(base_img_dir, row['image_filename'])
                        prompt = row['prompt']
                        print(f"  Processing {os.path.basename(img_path)} with prompt: {prompt}")
                        
                        input_img = Image.open(img_path).convert("RGB")
                        input_img = ImageOps.exif_transpose(input_img)
                        
                        ####
                        
                        original_size = input_img.size
                        print(f"    Original size: {original_size}")
                        
                        # Apply aspect ratio preservation if requested
                        resize_info = None
                        if args.preserve_aspect_ratio:
                            original_for_resize = input_img.copy()
                            input_img, resize_info = resize_with_aspect_ratio_padding(
                                input_img, 
                                max_size=args.max_dimension, 
                                pad_color=args.pad_color_rgb
                            )
                            print(f"    Resized for inpainting: {input_img.size} (with padding)")

                        print(f"    Prompt used: '{prompt}'")
                        
                        
                        ####
                        
                        # --- FIX 4: Resize images to max 1024 AND force multiples of 16 ---
                        # input_img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                        # w, h = input_img.size
                        # # Force divisible by 16 to avoid VAE/DiT patch crashing
                       
                        
                        pre_pipeline_size = input_img.size
                        w, h = pre_pipeline_size
                        safe_w = w - (w % 16)
                        safe_h = h - (h % 16)
                        
                        did_safe_resize = False
                        if (w, h) != (safe_w, safe_h):
                            # Facciamo un leggero downscale per evitare il crash dei VAE/DiT
                            input_img = input_img.resize((safe_w, safe_h), Image.Resampling.LANCZOS)
                            did_safe_resize = True
                            print(f"    [Safe Resize] Adapted to multiples of 16: {input_img.size}")
                        
                        print(f"  Input image size before pipeline: {input_img.size}")
                        
                        effective_steps, effective_guidance, effective_strength = get_quality_params(
                            model_key, 
                            args.steps, 
                            args.guidance, 
                            args.strength,
                        )
                        print(f"    Using steps: {effective_steps}, guidance: {effective_guidance}, strength: {effective_strength}")
                        
                        output_img = generator.generate(
                            prompt,
                            image=input_img,
                            strength=effective_strength,
                            steps=effective_steps,
                            guidance_scale=effective_guidance,
                            seed=seed
                        )
                        
                        if did_safe_resize:
                            output_img = output_img.resize(pre_pipeline_size, Image.Resampling.LANCZOS)
                            print(f"    [Safe Restore] Restored to original size: {output_img.size}")
                        
                        delta_raw = mean_abs_pixel_diff(input_img, output_img)
                        print(f"    Mean absolute pixel diff (raw output vs input): {delta_raw:.2f}")
                        
                        # Restore to original geometry after model generation
                        if resize_info is not None:
                            output_img, restore_mode = restore_output_to_original(output_img, resize_info)
                            print(f"    Restored output geometry ({restore_mode}): {output_img.size}")
                        
                        seed += 1
                        
                        base_name = os.path.splitext(os.path.basename(img_path))[0]
                        seed_str = f"_seed{seed-1}" if seed is not None else ""
                        out_name = f"{args.output_dir}/{base_name}_{model_key}_{quant}{seed_str}.png"
                        output_img.save(out_name)
                        print(f"    Saved {out_name}")
                        
                    except Exception as e_img:
                        print(f"    Failed to process {row.get('image_filename', 'unknown')}: {e_img}")
                
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