import argparse
import os
import glob
import numpy as np
from PIL import Image, ImageOps, ImageFilter

try:
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    import torch
except Exception:
    CLIPSegProcessor = None
    CLIPSegForImageSegmentation = None
    torch = None


class AutoMaskGenerator:
    def __init__(self, backend="clipseg", device="cpu"):
        self.backend = backend
        self.device = device
        self.processor = None
        self.model = None

        if self.backend != "clipseg":
            raise ValueError(f"Unsupported auto-mask backend: {self.backend}")

        if CLIPSegProcessor is None or CLIPSegForImageSegmentation is None:
            raise RuntimeError(
                "CLIPSeg backend requested but transformers is not available. "
                "Install with: pip install transformers torch"
            )

        print("Loading CLIPSeg for automatic mask generation...")
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", use_fast=True)
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        image,
        prompt,
        threshold=0.5,
        dilation=0,
        blur=0.0,
    ):
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        image = ImageOps.exif_transpose(image)

        inputs = self.processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.sigmoid(logits)[0].detach().float().cpu().numpy()
        binary = (probs >= threshold).astype(np.uint8) * 255
        mask_img = Image.fromarray(binary, mode="L").resize(image.size, Image.Resampling.BILINEAR)

        if dilation > 0:
            size = max(3, int(dilation))
            if size % 2 == 0:
                size += 1
            mask_img = mask_img.filter(ImageFilter.MaxFilter(size=size))

        if blur > 0:
            mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur))

        mask_array = np.array(mask_img)
        mask_img = Image.fromarray((mask_array >= 127).astype(np.uint8) * 255, mode="L")
        return mask_img


def load_prompts_file(filepath):
    """
    Load per-image prompts from CSV file.
    
    Format: filename,mask_prompt,inpaint_prompt
    Lines starting with # are ignored.
    
    Returns:
        dict mapping filenames (and basenames) to mask_prompt
    """
    prompts_map = {}
    print(f"Loading per-image prompts from: {filepath}")
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                if len(parts) >= 2:
                    img_name = parts[0].strip()
                    mask_prompt = parts[1].strip()
                    # Support both full filename and basename without extension
                    prompts_map[img_name] = mask_prompt
                    basename_no_ext = os.path.splitext(img_name)[0]
                    prompts_map[basename_no_ext] = mask_prompt
        print(f"  Loaded {len(prompts_map)} custom mask prompts")
    except Exception as e:
        print(f"  Warning: Failed to load prompts file: {e}")
    return prompts_map


def generate_masks(
    input_dir,
    output_dir,
    default_prompt="face",
    prompts_file=None,
    backend="clipseg",
    threshold=0.5,
    dilation=3,
    blur=0.0,
    device="cuda",
    max_images=None,
    overwrite=False,
):
    os.makedirs(output_dir, exist_ok=True)

    # Load per-image prompts if provided
    mask_prompts_map = {}
    if prompts_file:
        mask_prompts_map = load_prompts_file(prompts_file)

    # Find all images
    input_images = sorted(
        glob.glob(os.path.join(input_dir, '*.jpg')) + 
        glob.glob(os.path.join(input_dir, '*.png')) +
        glob.glob(os.path.join(input_dir, '*.PNG')) +
        glob.glob(os.path.join(input_dir, '*.JPG')) +
        glob.glob(os.path.join(input_dir, '*.jpeg'))
    )

    if not input_images:
        print(f"No images found in {input_dir}")
        return

    if max_images:
        input_images = input_images[:max_images]
        print(f"Processing max {max_images} images.")

    # Initialize generator
    generator = AutoMaskGenerator(backend=backend, device=device)

    print(f"Generating masks in: {output_dir}")
    print(f"Default mask prompt: '{default_prompt}'")
    print(f"Found {len(input_images)} images to process")

    for img_path in input_images:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        full_name = os.path.basename(img_path)
        out_mask_path = os.path.join(output_dir, f"{base_name}.png")

        if os.path.exists(out_mask_path) and not overwrite:
            print(f"  Skipping existing mask: {out_mask_path}")
            continue

        # Determine mask prompt for this image
        if full_name in mask_prompts_map:
            current_mask_prompt = mask_prompts_map[full_name]
        elif base_name in mask_prompts_map:
            current_mask_prompt = mask_prompts_map[base_name]
        else:
            current_mask_prompt = default_prompt

        try:
            input_img = Image.open(img_path).convert("RGB")
            input_img = ImageOps.exif_transpose(input_img)
            mask = generator.generate(
                input_img,
                prompt=current_mask_prompt,
                threshold=threshold,
                dilation=dilation,
                blur=blur,
            )
            mask.save(out_mask_path)
            print(f"  Saved: {out_mask_path} (prompt: '{current_mask_prompt}')")
        except Exception as e:
            print(f"  Failed to generate mask for {img_path}: {e}")

    print(f"\nDone! Generated masks saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate inpainting masks using CLIPSeg based on text prompts.")
    parser.add_argument("--input_dir",type=str,required=True,help="Directory containing input images")
    parser.add_argument("--output_dir",type=str,default="masks_auto",help="Directory to save generated masks (default: masks_auto)")
    parser.add_argument("--mask_prompt",type=str,default="face",help="Default text prompt for mask generation (default: face)")
    parser.add_argument("--mask_prompts_file",type=str,default=None,help="CSV file with per-image prompts (format: filename,mask_prompt,inpaint_prompt)")
    parser.add_argument("--backend",type=str,default="clipseg",choices=["clipseg"],help="Mask generation backend (default: clipseg)")
    parser.add_argument("--threshold",type=float,default=0.3,help="Threshold for mask binarization, 0.0-1.0 (default: 0.5)")
    parser.add_argument("--dilation",type=int,default=10,help="Mask dilation size in pixels, 0 to disable (default: 3)")
    parser.add_argument("--blur",type=float,default=5.0,help="Gaussian blur radius on mask edges, 0.0 to disable (default: 0.0)")
    parser.add_argument("--device",type=str,default=None,help="Device to use (cuda, cpu, mps). Auto-detected if not specified.")
    parser.add_argument("--max_images",type=int,default=None,help="Maximum number of images to process (default: all)")
    parser.add_argument("--overwrite",action="store_true",help="Overwrite existing masks")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist.")
        return
    
    # Auto-detect device
    if args.device is None:
        if torch and torch.cuda.is_available():
            args.device = "cuda"
        elif torch and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    
    print(f"Using device: {args.device}")
    
    generate_masks(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        default_prompt=args.mask_prompt,
        prompts_file=args.mask_prompts_file,
        backend=args.backend,
        threshold=args.threshold,
        dilation=args.dilation,
        blur=args.blur,
        device=args.device,
        max_images=args.max_images,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
