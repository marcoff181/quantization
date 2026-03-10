import argparse
import gc
import os
import random

import torch
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
    "sd3": "stabilityai/stable-diffusion-3-medium-diffusers",
    "sd35": "stabilityai/stable-diffusion-3.5-medium",
    "flux": "black-forest-labs/FLUX.1-dev",
    "sd15": "runwayml/stable-diffusion-v1-5",
    "z-image": "Tongyi-MAI/Z-Image",
    "flux2": "black-forest-labs/FLUX.2-klein-9B",
}
# High-quality defaults used when --steps/--guidance are not explicitly provided.
QUALITY_PRESETS = {
    "sd15": {"steps": 50, "guidance": 8.0},
    "sdxl": {"steps": 50, "guidance": 7.5},
    "sd3": {"steps": 45, "guidance": 6.0},
    "sd35": {"steps": 45, "guidance": 6.0},
    "flux": {"steps": 50, "guidance": 4.0},
    "z-image": {"steps": 50, "guidance": 4.0},
    "flux2": {"steps": 4, "guidance": 1.0},
}

# Conservative first-pass sharding support: large transformer-based pipelines.
SHARDED_SUPPORTED_MODELS = {"flux", "flux2", "sd3", "sd35"}
# Quantized bitsandbytes modules are routed away from auto-sharded mode by default.
SHARDED_UNSUPPORTED_QUANT = {"fp8", "fp4"}


def get_components(model_key):
    # Components selected here are the ones eligible for bitsandbytes quantization.
    if model_key == "sdxl":
        return ["text_encoder", "text_encoder_2", "unet"]
    if model_key == "sd15":
        return ["text_encoder", "unet"]
    if model_key == "flux":
        return ["text_encoder", "text_encoder_2", "transformer"]
    if model_key in ["sd3", "sd35"]:
        return ["text_encoder", "text_encoder_2", "text_encoder_3", "transformer"]
    if model_key in ["z-image", "flux2"]:
        return ["text_encoder", "transformer"]
    raise ValueError(f"Unknown model key: {model_key}")


def quantization_levels(model_key):
    # fp16 means unquantized weights at the selected torch dtype.
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
    # CLI overrides win; otherwise use model-tuned quality defaults.
    preset = QUALITY_PRESETS.get(model_key, {"steps": 50, "guidance": 7.0})
    steps = user_steps if user_steps is not None else preset["steps"]
    guidance = user_guidance if user_guidance is not None else preset["guidance"]
    return steps, guidance


def parse_max_memory(entries):
    if not entries:
        return None
    max_memory = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --max_memory entry '{entry}'. Expected format like 0=20GiB or cpu=64GiB."
            )
        key, value = entry.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        # Accelerate accepts integer GPU ids and the literal "cpu" key.
        if key == "cpu":
            max_memory["cpu"] = value
        else:
            max_memory[int(key)] = value
    return max_memory


def flush():
    # Free as much memory as possible between model runs.
    gc.collect()
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            with torch.cuda.device(idx):
                torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


class ImageGenerator:
    def __init__(
        self,
        model_key,
        quantization,
        device=None,
        monitor=None,
        device_map=None,
        max_memory=None,
        dtype_override="auto",
    ):
        self.model_key = model_key
        self.quantization = quantization
        self.monitor = monitor
        self.device_map = device_map
        self.max_memory = max_memory
        self.dtype_override = dtype_override
        # "sharded" means one model split across multiple visible GPUs via accelerate.
        self.sharded = self.device_map not in (None, "none")
        if self.sharded:
            if not torch.cuda.is_available():
                raise RuntimeError("device_map mode requires CUDA.")
            self.device = None
        else:
            if device is None:
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            else:
                self.device = device
        if self.sharded:
            print(f"Using sharded pipeline with device_map={self.device_map}")
            print(f"Visible CUDA devices: {torch.cuda.device_count()}")
            if self.max_memory:
                print(f"Max memory map: {self.max_memory}")
        else:
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
        # Explicit dtype override applies only to sharded mode for predictable placement.
        if self.sharded and self.dtype_override == "fp16":
            return torch.float16
        if self.sharded and self.dtype_override == "bf16":
            if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
                raise RuntimeError(
                    "--dtype bf16 requested for sharded mode, but bf16 is not supported by this CUDA setup."
                )
            return torch.bfloat16

        if not self.sharded and self.device == "cpu":
            return torch.float32
        if self.model_key in ["flux", "sd3", "sd35", "z-image", "flux2"]:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float16 if torch.cuda.is_available() or self.sharded else torch.float32
    
    def _build_common_load_kwargs(self, dtype, q_config):
        # Shared kwargs for every pipeline class, with optional sharding knobs.
        kwargs = {"torch_dtype": dtype}
        if q_config is not None:
            kwargs["quantization_config"] = q_config
        if self.sharded:
            kwargs["device_map"] = self.device_map
            if self.max_memory:
                kwargs["max_memory"] = self.max_memory
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

        if self.model_key == "flux2":
            return Flux2KleinPipeline.from_pretrained(
                model_id,
                **common_kwargs,
            )

        variant = (
            "fp16"
            if (self.quantization == "fp16" and not self.sharded and self.device != "cpu")
            else None
        )
        extra_kwargs = dict(common_kwargs)
        extra_kwargs["use_safetensors"] = True
        if variant is not None:
            extra_kwargs["variant"] = variant
        return AutoPipelineForText2Image.from_pretrained(
            model_id,
            **extra_kwargs,
        )

    def _configure_pipeline_device(self):
        if self.sharded:
            # Sharded mode relies on accelerate placement; avoid manual moves/offload hooks.
            print("  Loaded with device map; skipping .to(...) and CPU offload.")
            if hasattr(self.pipeline_t2i, "hf_device_map"):
                print(f"  hf_device_map: {self.pipeline_t2i.hf_device_map}")
            return

        if self.quantization == "fp8":
            try:
                self.pipeline_t2i.to(self.device)
                print(f"  Moved fp8 pipeline to {self.device}")
            except Exception:
                print(
                    f"  Warning: could not move fp8 pipeline to {self.device}, already on device"
                )
            return

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
    parser.add_argument(
        "--device_map",
        type=str,
        default="none",
        choices=["none", "balanced", "auto"],
        help="Shard a single pipeline across visible GPUs. Use with CUDA_VISIBLE_DEVICES=0,1 etc.",
    )
    parser.add_argument(
        "--max_memory",
        nargs="*",
        default=None,
        help="Optional max memory map, e.g. --max_memory 0=20GiB 1=20GiB cpu=64GiB",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "fp16", "bf16"],
        help="Dtype override for sharded mode. Use fp16/bf16 for predictable placement.",
    )
    parser.add_argument(
        "--sharded_quant_fallback",
        type=str,
        default="reroute",
        choices=["reroute", "skip", "error"],
        help=(
            "Behavior for fp8/fp4 when --device_map is enabled: "
            "reroute to non-sharded single-device run, skip, or error."
        ),
    )
    return parser


def validate_runtime_args(args):
    # Guardrails for mutually exclusive execution modes.
    if args.device_map != "none" and args.device is not None:
        raise ValueError("--device and --device_map are mutually exclusive.")

    if args.device_map != "none" and not torch.cuda.is_available():
        raise RuntimeError("--device_map requires CUDA.")

    if args.device_map != "none" and torch.cuda.device_count() < 2:
        print("Warning: --device_map was requested but fewer than 2 visible CUDA devices were found.")

    if args.device_map != "none":
        unsupported = [m for m in args.models if m not in SHARDED_SUPPORTED_MODELS]
        if unsupported:
            raise ValueError(
                "Sharded mode currently supports only models "
                f"{sorted(SHARDED_SUPPORTED_MODELS)}. Unsupported: {unsupported}"
            )


def get_reroute_device(args):
    # Preferred single-device target for rerouted quantized runs.
    if args.device:
        return args.device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_run_plan(args):
    # Build a per-quantization execution plan so one invocation can mix modes safely.
    plan = []
    sharded_mode = args.device_map != "none"
    reroute_device = get_reroute_device(args)

    for quant in args.quantization:
        if sharded_mode and quant in SHARDED_UNSUPPORTED_QUANT:
            if args.sharded_quant_fallback == "error":
                raise ValueError(
                    "Sharded mode only supports fp16/bf16 loading right now. "
                    f"Unsupported quantization in sharded mode: {quant}"
                )

            if args.sharded_quant_fallback == "skip":
                print(
                    f"Skipping quantization '{quant}' in sharded mode "
                    "because --sharded_quant_fallback=skip."
                )
                continue

            # "reroute": keep one-command UX by running this quant level non-sharded.
            print(
                f"Rerouting quantization '{quant}' to non-sharded execution on device {reroute_device}."
            )
            plan.append(
                {
                    "quant": quant,
                    "device_map": "none",
                    "device": reroute_device,
                    "dtype_override": "auto",
                }
            )
            continue

        plan.append(
            {
                "quant": quant,
                "device_map": args.device_map,
                "device": args.device,
                "dtype_override": args.dtype,
            }
        )

    if not plan:
        raise ValueError("No runnable quantization levels remain after applying fallback policy.")

    return plan


def resolve_monitor_device(args):
    # Metrics collector currently tracks one device type per invocation.
    if args.device_map != "none":
        return "cuda"
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
    # run_cfg is per-quantization execution policy (sharded vs rerouted device mode).
    quant = run_cfg["quant"]
    seed = args.seed
    generator = ImageGenerator(
        model_key,
        quant,
        device=run_cfg["device"],
        monitor=monitor,
        device_map=run_cfg["device_map"],
        max_memory=parsed_max_memory,
        dtype_override=run_cfg["dtype_override"],
    )

    effective_steps, effective_guidance = get_quality_params(
        model_key,
        args.steps,
        args.guidance,
    )
    print(f"Generating txt2img with {model_key} ({quant})...")
    print(
        f"Quality params -> steps: {effective_steps}, guidance: {effective_guidance}"
    )
    print(f"Generating {len(prompts)} images using {model_key} ({quant})...")

    for i, prompt in enumerate(prompts):
        img = generator.generate(
            prompt,
            steps=effective_steps,
            guidance_scale=effective_guidance,
            seed=seed,
        )
        seed += 1

        if img is None:
            print(
                f"Generation failed for model {model_key} with quantization {quant} "
                f"on prompt {i + 1}/{len(prompts)}: '{prompt}'"
            )
            continue

        out_name = (
            f"{args.output_dir}/{model_key}_{quant}_p{i}_seed{seed - 1}.png"
        )
        if os.path.exists(out_name):
            print(f"[{i + 1}/{len(prompts)}] File {out_name} already exists, replacing...")
        else:
            print(f"[{i + 1}/{len(prompts)}] Saved image {out_name}")
        img.save(out_name)

    del generator
    flush()


def main():
    parser = build_parser()
    args = parser.parse_args()

    validate_runtime_args(args)
    run_plan = build_run_plan(args)

    parsed_max_memory = parse_max_memory(args.max_memory)
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
                    parsed_max_memory,
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