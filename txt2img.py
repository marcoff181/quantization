import argparse
import gc
import os
import random
import time  # <-- Added for the sleep loop

import torch
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    FluxPipeline,
    ZImagePipeline,
    DiffusionPipeline,
)
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


# Conservative first-pass sharding support: large transformer-based pipelines.
SHARDED_SUPPORTED_MODELS = {"flux", "sd3", "sd35", "pg25"}
# Quantized bitsandbytes modules are routed away from auto-sharded mode by default.
SHARDED_UNSUPPORTED_QUANT = {"fp8", "fp4"}


class ImageGenerator:
    def __init__(
        self,
        model_key,
        quantization,
        device=None,
        monitor=None,
        max_memory=None,
        dtype_override="auto",
    ):
        self.model_key = model_key
        self.quantization = quantization
        self.monitor = monitor
        self.dtype_override = dtype_override
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
        # Load eagerly so per-model load cost is captured in monitor metrics.
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
        if self.model_key in ["flux", "sdxl", "sd3", "sd35", "z-image"]:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float16 if torch.cuda.is_available() else torch.float32
    
    def _build_common_load_kwargs(self, dtype, q_config):
        # Shared kwargs for every pipeline class.
        kwargs = {"torch_dtype": dtype}
        if q_config is not None:
            kwargs["quantization_config"] = q_config
        return kwargs

    def _load_text2image_pipeline(self, model_id, common_kwargs):
        # Select the correct pipeline class per model family.
        if self.model_key == "flux":
            return FluxPipeline.from_pretrained(
                model_id,
                **common_kwargs,
            )

        if self.model_key in ["sd35", "sd3"]:
            return AutoPipelineForText2Image.from_pretrained(
                model_id,
                **common_kwargs,
            )

        if self.model_key == "z-image":
            return ZImagePipeline.from_pretrained(
                model_id,
                **common_kwargs,
            )

        # THE FIX: Always use the fp16 variant on GPU to prevent FP16/FP32 bias collisions 
        variant = "fp16" if self.device != "cpu" else None
        
        extra_kwargs = dict(common_kwargs)
        extra_kwargs["use_safetensors"] = True
        
        if variant is not None:
            extra_kwargs["variant"] = variant
            
        return AutoPipelineForText2Image.from_pretrained(
            model_id,
            **extra_kwargs,
        )

    def _configure_pipeline_device(self):
        try:
            self.pipeline_t2i.to(self.device)
            print(f"  Moved pipeline to {self.device}")
        except Exception:
            print(f"  Warning: could not move pipeline to {self.device}, already on device")

    def _build_img2img_pipeline(self):
        # Best-effort conversion: txt2img is primary, img2img is optional.
        try:
            self.pipeline_i2i = AutoPipelineForImage2Image.from_pipe(self.pipeline_t2i)
        except Exception as e:
            print(f"  Warning: could not create img2img pipeline from pipe: {e}")
            self.pipeline_i2i = None

    def load_pipeline(self):
        # Centralized model load path so all modes share logging/metrics behavior.
        print(f"Loading {self.model_key} with {self.quantization} quantization...")
        model_id = MODELS[self.model_key]
        q_config = quantization_levels(self.model_key)[self.quantization]
        dtype = self._get_dtype()
        try:
            common_kwargs = self._build_common_load_kwargs(dtype, q_config)
            self.pipeline_t2i = self._load_text2image_pipeline(model_id, common_kwargs)
            self.model_size_mb = self._compute_model_size(self.pipeline_t2i)
            print(f"  Model parameter size: {self.model_size_mb:.1f} MB")
            self._configure_pipeline_device()
            self._build_img2img_pipeline()
        except Exception as e:
            print(f"Error loading pipeline: {e}")
            raise

    @staticmethod
    def _compute_model_size(pipeline):
        # Deduplicate shared tensors when estimating parameter memory footprint.
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
                if self.pipeline_i2i is None:
                    raise RuntimeError("img2img pipeline is not available for this configuration.")
                kwargs["image"] = image
                kwargs["strength"] = strength
                result = self.pipeline_i2i(**kwargs).images[0]
            else:
                result = self.pipeline_t2i(**kwargs).images[0]
            if self.monitor:
                self.monitor.record(
                    self.model_key,
                    self.quantization,
                    "generate",
                    extra={
                        "seed": seed,
                        "steps": steps,
                        "model_size_mb": self.model_size_mb,
                    },
                )
            return result
        except Exception as e:
            # Bubble up OOM errors so the main loop can sleep and retry
            if is_oom_error(e):
                raise e
            print(f"Text-to-Image not supported for this model configuration: {e}")
            return None


def build_parser():
    # Keep all CLI wiring in one place to simplify future option changes.
    parser = argparse.ArgumentParser(
        description="Generate images using multiple diffusion models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        choices=MODELS.keys(),
        help="Models to use. Default is all available models",
    )
    parser.add_argument("--prompt", type=str, default="", help="Prompt for generation.")
    parser.add_argument(
        "--prompts",
        type=int,
        default=100,
        help="If --prompt is not provided, how many prompts to load automatically",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="prompts_filtered.txt",
        help="File to load prompts from if --prompt is not provided",
    )
    parser.add_argument(
        "--quantization",
        nargs="+",
        default=list(quantization_levels("sd35").keys()),
        choices=quantization_levels("sd35").keys(),
        help="Quantization levels.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/media/SSD_4TB/crispy_storage",
        help="Directory for output images.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.3,
        help="Strength for img2img only (not used in txt2img mode).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Inference steps. If omitted, uses high-quality model defaults.",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=None,
        help="Guidance scale. If omitted, uses high-quality model defaults.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device to use.")
    parser.add_argument(
        "--seed", type=int, default=123, help="Fixed seed for reproducibility."
    )
    return parser


def get_reroute_device(args):
    if args.device:
        return args.device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_run_plan(args):
    # Build a per-quantization execution plan for single device only.
    plan = []
    for quant in args.quantization:
        plan.append(
            {
                "quant": quant,
                "device": args.device,
                "dtype_override": "auto",
            }
        )
    if not plan:
        raise ValueError("No runnable quantization levels remain after applying fallback policy.")
    return plan


def resolve_monitor_device(args):
    # Metrics collector currently tracks one device type per invocation.
    if args.device:
        return args.device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_prompts(args):
    # Prompt source priority: explicit --prompt, else top-N prompts from file.
    if args.prompt == "" and args.prompts > 0:
        print(f"Loading prompts from {args.prompts_file}...")
        with open(args.prompts_file, "r", encoding="utf-8") as file:
            prompts = [line.strip() for line in file if line.strip()]
        if len(prompts) < args.prompts:
            print(
                f"Warning: requested {args.prompts} prompts but only found {len(prompts)} in file. Using all available prompts."
            )
            return prompts
        return prompts[: args.prompts]

    return [args.prompt]


def generate_for_config(args, monitor, parsed_max_memory, model_key, run_cfg, prompts):
    quant = run_cfg["quant"]
    seed = args.seed
    
    generator = None
    # 1. RETRY LOOP FOR MODEL LOADING (OOM can happen here when VRAM is stolen)
    while True:
        try:
            generator = ImageGenerator(
                model_key,
                quant,
                device=run_cfg["device"],
                monitor=monitor,
                max_memory=parsed_max_memory,
                dtype_override=run_cfg["dtype_override"],
            )
            break  # Exit loop if successfully loaded
        except Exception as e:
            if is_oom_error(e):
                print(f"[{model_key} - {quant}] GPU OOM during model load! Waiting 60 seconds...")
                flush()
                time.sleep(60)  # Sleep idly and retry
            else:
                raise e  # Bubble up if it's a completely different error

    effective_steps, effective_guidance, _ = get_quality_params(
        model_key,
        args.steps,
        args.guidance,
    )
    print(f"Generating txt2img with {model_key} ({quant})...")
    print(f"Quality params -> steps: {effective_steps}, guidance: {effective_guidance}")
    print(f"Generating {len(prompts)} images using {model_key} ({quant})...")

    # 2. RETRY LOOP FOR IMAGE GENERATION

    i = 0
    while i < len(prompts):
        prompt = prompts[i]
        out_name = f"{args.output_dir}/{model_key}_{quant}_p{i}_seed{seed}.png"

        # Skip generation if output file already exists
        if os.path.exists(out_name):
            print(f"[{i + 1}/{len(prompts)}] File {out_name} already exists, skipping...")
            i += 1
            seed += 1
            continue

        try:
            img = generator.generate(
                prompt,
                steps=effective_steps,
                guidance_scale=effective_guidance,
                seed=seed,
            )

            if img is not None:
                img.save(out_name)
                print(f"[{i + 1}/{len(prompts)}] Saved image {out_name}")
            else:
                print(f"[{i + 1}/{len(prompts)}] Generation failed (non-OOM error) for prompt: '{prompt}'")

            # Move on to the next prompt upon success or a normal error
            i += 1
            seed += 1

        except Exception as e:
            if is_oom_error(e):
                print(f"[{model_key} - {quant}] GPU OOM during generation! Waiting 60 seconds...")
                flush()
                time.sleep(60)
                # Notice we DO NOT increment 'i' or 'seed' here, so the while loop tries the EXACT same image again.
            else:
                print(f"Fatal generation error: {e}")
                # Move past it if it's completely broken so the whole batch doesn't stall forever
                i += 1
                seed += 1

    del generator
    flush()


def main():
    parser = build_parser()
    args = parser.parse_args()

    run_plan = build_run_plan(args)

    os.makedirs(args.output_dir, exist_ok=True)

    prompts = load_prompts(args)

    monitor_device = resolve_monitor_device(args)
    monitor = ResourceMonitor(device=monitor_device)

    for model_key in args.models:
        for run_cfg in run_plan:
            try:
                generate_for_config(
                    args,
                    monitor,
                    None,  
                    model_key,
                    run_cfg,
                    prompts,
                )
            except Exception as e:
                print(f"Failed to run {model_key} with {run_cfg['quant']}: {e}")
                flush()

    metrics_path = os.path.join(args.output_dir, "resource_metrics.csv")
    monitor.save_csv(metrics_path)
    monitor.summary()


if __name__ == "__main__":
    main()