import argparse
import os
import glob
import random
import torch
import gc
import numpy as np
from PIL import Image, ImageOps

from diffusers import (
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
)

from diffusers.quantizers import PipelineQuantizationConfig
from resource_monitor import ResourceMonitor

from shared_utils import (
    MODELS_INPAINT, 
    flush, 
    quantization_levels, 
    load_prompts_file
)

# --- Configuration & Constants ---


def resize_with_aspect_ratio_padding(image, max_size=512, base_unit=64, pad_color=(128, 128, 128)):
    """
    Resize image preserving aspect ratio and pad to multiple of base_unit.
    
    Args:
        image: PIL Image
        max_size: Maximum size of the longer sideMODELS (default 512)
        base_unit: Pad to multiple of this value (default 64, required by diffusion MODELS_INPAINT)
        pad_color: RGB tuple for padding color (default gray: 128,128,128)
    
    Returns:
        (padded_image, resize_info) tuple where resize_info stores geometry
    """
    original_size = image.size  # (width, height)
    
    # Calculate resize dimensions preserving aspect ratio
    width, height = original_size
    scale = min(max_size / width, max_size / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Pad to multiple of base_unit
    pad_width = ((new_width + base_unit - 1) // base_unit) * base_unit
    pad_height = ((new_height + base_unit - 1) // base_unit) * base_unit
    
    # Create padded image with gray background
    padded = Image.new("RGB", (pad_width, pad_height), pad_color)
    # Center the resized image
    offset_x = (pad_width - new_width) // 2
    offset_y = (pad_height - new_height) // 2
    padded.paste(resized, (offset_x, offset_y))
    
    resize_info = {
        "original_size": original_size,
        "new_size": (new_width, new_height),
        "offset": (offset_x, offset_y),
        "padded_size": (pad_width, pad_height),
        "did_resize": (new_width, new_height) != original_size,
        "did_padding": (pad_width, pad_height) != (new_width, new_height),
    }
    return padded, resize_info


def resize_mask_with_padding(mask_image, resize_info):
    """
    Apply the exact same resize+padding transform to a mask.

    Args:
        mask_image: PIL mask image (mode L)
        resize_info: dict produced by resize_with_aspect_ratio_padding

    Returns:
        Padded mask image aligned with padded input image
    """
    new_width, new_height = resize_info["new_size"]
    offset_x, offset_y = resize_info["offset"]
    pad_width, pad_height = resize_info["padded_size"]

    resized_mask = mask_image.resize((new_width, new_height), Image.Resampling.NEAREST)
    padded_mask = Image.new("L", (pad_width, pad_height), 0)
    padded_mask.paste(resized_mask, (offset_x, offset_y))
    return padded_mask


def extract_unpadded_region(image, resize_info):
    new_width, new_height = resize_info["new_size"]
    offset_x, offset_y = resize_info["offset"]
    return image.crop((offset_x, offset_y, offset_x + new_width, offset_y + new_height))


def restore_output_to_original(image, resize_info):
    """
    Restore model output back to original geometry robustly.

    Returns:
        (restored_image, restore_mode)
    """
    original_size = resize_info["original_size"]
    padded_size = resize_info["padded_size"]
    did_resize = resize_info["did_resize"]
    did_padding = resize_info["did_padding"]

    if image.size != padded_size:
        restored = image.resize(original_size, Image.Resampling.LANCZOS)
        return restored, "fallback_resize_mismatch"

    if not did_resize and not did_padding:
        return image, "identity"

    cropped = extract_unpadded_region(image, resize_info)
    restored = cropped.resize(original_size, Image.Resampling.LANCZOS)
    return restored, "crop_and_resize"


def crop_to_original(image, resize_info):
    """
    Crop padded image back to original dimensions.
    
    Args:
        image: Padded PIL Image
        resize_info: dict from resize_with_aspect_ratio_padding
    
    Returns:
        Cropped PIL Image resized back to original image dimensions
    """
    restored, _ = restore_output_to_original(image, resize_info)
    return restored


def mask_white_ratio(mask_image):
    mask_arr = np.array(mask_image, dtype=np.uint8)
    return float((mask_arr > 127).mean())


def mean_abs_pixel_diff(img_a, img_b):
    if img_a.size != img_b.size:
        img_b = img_b.resize(img_a.size, Image.Resampling.LANCZOS)
    arr_a = np.array(img_a.convert("RGB"), dtype=np.float32)
    arr_b = np.array(img_b.convert("RGB"), dtype=np.float32)
    return float(np.mean(np.abs(arr_a - arr_b)))


def create_mask(image_input, mask_type="center_circle", mask_size_ratio=0.5):
    """
    Create an inpainting mask for the given image.
    
    Args:
        image_input: Path to input image or a PIL image
        mask_type: "center_circle", "center_square", "random_region", "full"
        mask_size_ratio: Proportion of image to mask (0.0-1.0)
    
    Returns:
        PIL Image with white regions to inpaint, black regions to preserve
    """
    if isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        img = Image.open(image_input).convert("RGB")
    width, height = img.size
    
    mask = Image.new("L", (width, height), 0)  # Black canvas (preserve)
    
    if mask_type == "center_circle":
        radius = int(min(width, height) * mask_size_ratio / 2)
        cx, cy = width // 2, height // 2
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=255
        )
    elif mask_type == "center_square":
        size = int(min(width, height) * mask_size_ratio)
        x0 = (width - size) // 2
        y0 = (height - size) // 2
        mask.paste(255, (x0, y0, x0 + size, y0 + size))
    elif mask_type == "random_region":
        size = int(min(width, height) * mask_size_ratio)
        x0 = random.randint(0, max(0, width - size))
        y0 = random.randint(0, max(0, height - size))
        mask.paste(255, (x0, y0, min(x0 + size, width), min(y0 + size, height)))
    elif mask_type == "full":
        mask = Image.new("L", (width, height), 255)  # All white (inpaint everywhere)
    
    return mask


class InpaintGenerator:
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

        self.pipeline_inpaint = None
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
        print(f"Loading {self.model_key} with {self.quantization} quantization (Inpainting)...")
        
        model_id = MODELS_INPAINT[self.model_key]
        q_config = quantization_levels(self.model_key)[self.quantization]
        dtype = self._get_dtype()
        
        try:
            # Try inpainting pipeline
            try:
                if self.model_key in ["sd3", "sd35"]:
                    self.pipeline_inpaint = AutoPipelineForInpainting.from_pretrained(
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
                    self.pipeline_inpaint = AutoPipelineForInpainting.from_pretrained(
                        model_id,
                        quantization_config=q_config,
                        torch_dtype=dtype,
                        variant=variant,
                        use_safetensors=True,
                    )
            except Exception as e:
                print(f"  Warning: Inpainting not available for {self.model_key}, falling back to Text2Image: {e}")
                # Fallback: load text2image and convert
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
                # Try to convert to inpainting
                try:
                    self.pipeline_inpaint = AutoPipelineForInpainting.from_pipe(self.pipeline_t2i)
                except Exception as e2:
                    raise RuntimeError(f"Could not create inpainting pipeline: {e2}")

            self.model_size_mb = self._compute_model_size(self.pipeline_inpaint)
            print(f"  Model parameter size: {self.model_size_mb:.1f} MB")

            if self.quantization == "fp8":
                try:
                    self.pipeline_inpaint.to(self.device)
                    print(f"  Moved fp8 pipeline to {self.device}")
                except Exception:
                    print(f"  Warning: could not move fp8 pipeline to {self.device}, already on device")
            else:
                try:
                    self.pipeline_inpaint.to(self.device)
                    print(f"  Moved pipeline to {self.device}")
                except Exception as e:
                    print(f"  Warning: could not move pipeline to {self.device}: {e}. It may already be on the device or not support .to()")

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

    def generate(self, prompt, image, mask, strength=0.8, guidance_scale=7.5, steps=30, seed=None):
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator("cpu").manual_seed(seed)

        print(f"Generating with seed: {seed}")

        if self.monitor:
            self.monitor.start_timer()

        kwargs = {
            "prompt": prompt,
            "image": image,
            "mask_image": mask,
            "strength": strength,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator 
        }
    
        result = self.pipeline_inpaint(**kwargs).images[0]
        if self.monitor:
            self.monitor.record(
                self.model_key,
                self.quantization,
                "generate",
                extra={"seed": seed, "steps": steps, "model_size_mb": self.model_size_mb},
            )
            
        print(f"  Generated image size: {result.size}")
        print(f"  Generation complete for prompt: {prompt[:50]}...")
        return result

def main():
    parser = argparse.ArgumentParser(description="Inpaint images using various Stable Diffusion MODELS_INPAINT.")
    parser.add_argument("--prompts_file", type=str, required=True, help="CSV file with per-image prompts (format: filename,mask_prompt,inpaint_prompt).")
    parser.add_argument("--MODELS_INPAINT", nargs='+', default=['sd3'], choices=MODELS_INPAINT.keys(), help="MODELS_INPAINT to use.")
    parser.add_argument("--quantization", nargs='+', default=['fp16'], choices=quantization_levels("sd35").keys(), help="Quantization levels.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images.")
    parser.add_argument("--output_dir", type=str, default="out/inpaint", help="Directory for output images.")
    parser.add_argument("--device", type=str, default=None, help="Device to use.")
    parser.add_argument("--seed", type=int, default=123, help="Fixed seed for reproducibility.")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum number of images to process (default: all).")
    
    parser.add_argument("--mask_dir", type=str, default=None, help="Directory containing mask images (same names as inputs). If not provided, geometric masks are auto-generated. Use generate_masks.py for CLIPSeg masks.")
    parser.add_argument("--mask_type", type=str, default="center_circle", choices=["center_circle", "center_square", "random_region", "full"], help="Type of geometric mask to auto-generate if --mask_dir not provided.")
    parser.add_argument("--mask_size_ratio", type=float, default=0.2, help="Proportion of image to inpaint (0.0-1.0) for geometric masks.")
    
    parser.add_argument("--strength", type=float, default=0.5, help="Inpainting strength (0.0 to 1.0).")
    parser.add_argument("--steps", type=int, default=25, help="Inference steps.")
    parser.add_argument("--guidance", type=float, default=4.0, help="Guidance scale.")
    
    parser.add_argument("--preserve_aspect_ratio", action="store_true", help="Preserve image aspect ratio with smart padding instead of resizing to square.")
    parser.add_argument("--max_dimension", type=int, default=512, help="Maximum dimension when preserving aspect ratio (default: 512). Longer side won't exceed this.")
    parser.add_argument("--pad_color", type=str, default="128,128,128", help="RGB padding color as 'R,G,B' when preserving aspect ratio (default: 128,128,128 gray).")
    
    args = parser.parse_args()
    
    # Parse pad_color
    try:
        pad_colors = args.pad_color.split(',')
        args.pad_color_rgb = tuple(int(c.strip()) for c in pad_colors)
        if len(args.pad_color_rgb) != 3 or any(c < 0 or c > 255 for c in args.pad_color_rgb):
            raise ValueError
    except:
        print(f"Invalid pad_color: {args.pad_color}. Using default 128,128,128")
        args.pad_color_rgb = (128, 128, 128)
    
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

    if args.max_images:
        input_images = input_images[:args.max_images]
        print(f"Processing max {args.max_images} images.")

    # Load per-image prompts from CSV
    _, inpaint_prompts_map = load_prompts_file(args.prompts_file)

    for model_key in args.MODELS_INPAINT:
        for quant in args.quantization:
            seed = args.seed
            try:
                generator = InpaintGenerator(model_key, quant, device=args.device, monitor=monitor)
                
                print(f"Inpainting with {model_key} ({quant})...")
                
                for i, img_path in enumerate(input_images):
                    try:
                        print(f"  Processing {os.path.basename(img_path)}...")
                        base_name = os.path.splitext(os.path.basename(img_path))[0]
                        debug_step_idx = 0
                        step_dir = None
                        
                        # Load and orient image
                        input_img = Image.open(img_path).convert("RGB")
                        input_img = ImageOps.exif_transpose(input_img)
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
                        
                        # Determine inpainting prompt for this image
                        full_name = os.path.basename(img_path)
                        if full_name in inpaint_prompts_map:
                            current_prompt = inpaint_prompts_map[full_name]
                        elif base_name in inpaint_prompts_map:
                            current_prompt = inpaint_prompts_map[base_name]
                        else:
                            print(f"    Warning: image '{base_name}' not found in CSV. Skipping...")
                            continue

                        print(f"    Prompt used: '{current_prompt}'")
                        if not current_prompt or not current_prompt.strip():
                            print("    Warning: empty prompt in CSV. Inpainting changes may be very small.")
                        
                        # Load or generate mask
                        if args.mask_dir:
                            base_name = os.path.splitext(os.path.basename(img_path))[0]
                            mask_path = os.path.join(args.mask_dir, f"{base_name}.png")
                            if os.path.exists(mask_path):
                                mask_img = Image.open(mask_path).convert("L")
                                mask_img = ImageOps.exif_transpose(mask_img)
                                
                                if args.preserve_aspect_ratio and resize_info is not None:
                                    mask_img = resize_mask_with_padding(mask_img, resize_info)
                                elif mask_img.size != input_img.size:
                                    mask_img = mask_img.resize(input_img.size, Image.Resampling.NEAREST)
                                print(f"    Loaded mask: {mask_path}")
                            else:
                                print(f"    Mask not found: {mask_path}, auto-generating...")
                                mask_img = create_mask(input_img, args.mask_type, args.mask_size_ratio)
                        else:
                            mask_img = create_mask(input_img, args.mask_type, args.mask_size_ratio)

                        white_ratio = mask_white_ratio(mask_img)
                        print(f"    Mask white coverage: {white_ratio*100:.2f}%")
                        if white_ratio < 0.005:
                            print("    Warning: mask is almost empty. Prompt effect will be minimal.")
                        elif white_ratio > 0.995:
                            print("    Warning: mask is almost full image. Changes may affect everything.")
                        
                        output_img = generator.generate(
                            current_prompt, 
                            image=input_img,
                            mask=mask_img,
                            strength=args.strength,
                            steps=args.steps, 
                            guidance_scale=args.guidance, 
                            seed=seed
                        )

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
                        print(f"    Saved {out_name} ({output_img.size})")
                        
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
