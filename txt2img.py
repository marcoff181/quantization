import argparse
import csv
import os
import random
import time
import torch
import gc

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not installed. RAM monitoring disabled. Install with: pip install psutil")

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


class ResourceMonitor:
    """Monitors GPU VRAM, system RAM, and timing during pipeline operations."""

    def __init__(self, device="cuda"):
        self.device = device
        self.records = []  # list of dicts for CSV export
        self._use_cuda = device == "cuda" and torch.cuda.is_available()

    # ---- snapshot helpers ----
    def _gpu_mem_mb(self):
        if not self._use_cuda:
            return 0.0, 0.0, 0.0
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        return allocated, reserved, peak

    def _ram_mb(self):
        if not HAS_PSUTIL:
            return 0.0, 0.0
        proc = psutil.Process(os.getpid())
        rss = proc.memory_info().rss / 1024**2
        sys_used = psutil.virtual_memory().used / 1024**2
        return rss, sys_used

    def _gpu_total_mb(self):
        if not self._use_cuda:
            return 0.0
        return torch.cuda.get_device_properties(0).total_memory / 1024**2

    # ---- context-manager style helpers ----
    def reset_peak(self):
        if self._use_cuda:
            torch.cuda.reset_peak_memory_stats()

    def start_timer(self):
        self._t0 = time.perf_counter()
        self.reset_peak()
        self._start_alloc, _, _ = self._gpu_mem_mb()
        self._start_rss, _ = self._ram_mb()

    def stop_timer(self):
        elapsed = time.perf_counter() - self._t0
        alloc, reserved, peak = self._gpu_mem_mb()
        rss, sys_used = self._ram_mb()
        return {
            "elapsed_s": round(elapsed, 2),
            "vram_allocated_mb": round(alloc, 1),
            "vram_reserved_mb": round(reserved, 1),
            "vram_peak_mb": round(peak, 1),
            "vram_total_mb": round(self._gpu_total_mb(), 1),
            "ram_process_mb": round(rss, 1),
            "ram_system_used_mb": round(sys_used, 1),
            "ram_delta_mb": round(rss - self._start_rss, 1),
        }

    # ---- high-level API ----
    def record(self, model_key, quantization, phase, extra=None):
        """Stop timer and store a record with metadata."""
        metrics = self.stop_timer()
        metrics.update({"model": model_key, "quantization": quantization, "phase": phase})
        if extra:
            metrics.update(extra)
        self.records.append(metrics)
        self._print_record(metrics)
        return metrics

    def _print_record(self, m):
        print(f"  [{m['phase']}] {m['model']}@{m['quantization']}  "
              f"time={m['elapsed_s']}s  "
              f"VRAM peak={m['vram_peak_mb']}MB / {m['vram_total_mb']}MB  "
              f"RAM={m['ram_process_mb']}MB")

    def save_csv(self, path):
        """Write all collected records to a CSV file."""
        if not self.records:
            return
        fieldnames = list(self.records[0].keys())
        # ensure all keys are captured
        for r in self.records:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)
        print(f"\nResource metrics saved to {path}")

    def summary(self):
        """Print a summary table of all records."""
        if not self.records:
            return
        print("\n" + "=" * 90)
        print(f"{'Model':<10} {'Quant':<6} {'Phase':<12} {'Time(s)':<9} "
              f"{'VRAM Peak(MB)':<15} {'RAM(MB)':<10}")
        print("-" * 90)
        for m in self.records:
            print(f"{m['model']:<10} {m['quantization']:<6} {m['phase']:<12} "
                  f"{m['elapsed_s']:<9} {m['vram_peak_mb']:<15} {m['ram_process_mb']:<10}")
        print("=" * 90)


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

        if self.monitor:
            self.monitor.start_timer()
        self.load_pipeline()
        if self.monitor:
            self.monitor.record(self.model_key, self.quantization, "load_model")

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
                    dtype=dtype,
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
                    extra={"seed": seed, "steps": steps}
                )
            return result
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
            try:
                generator = ImageGenerator(model_key, quant, device=args.device, monitor=monitor)

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

    # Save metrics and print summary
    metrics_path = os.path.join(args.output_dir, "resource_metrics.csv")
    monitor.save_csv(metrics_path)
    monitor.summary()


if __name__ == "__main__":
    main()
