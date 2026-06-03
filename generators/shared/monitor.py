import csv
import os
import time
import torch
import psutil
 
class ResourceMonitor:
    """Monitors GPU VRAM, system RAM, and timing during pipeline operations."""

    def __init__(self, device="cuda"):
        self.device = device
        self.records = []  # list of dicts for CSV export
        self._use_cuda = device == "cuda" and torch.cuda.is_available()

    def _gpu_mem_mb(self):
        if not self._use_cuda:
            return 0.0, 0.0, 0.0
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        return allocated, reserved, peak

    def _ram_mb(self):
        proc = psutil.Process(os.getpid())
        rss = proc.memory_info().rss / 1024**2
        sys_used = psutil.virtual_memory().used / 1024**2
        return rss, sys_used

    def _gpu_total_mb(self):
        if not self._use_cuda:
            return 0.0
        return torch.cuda.get_device_properties(0).total_memory / 1024**2

    def _reset_peak(self):
        if self._use_cuda:
            torch.cuda.reset_peak_memory_stats()

    def start_timer(self):
        """Start timing and resource tracking for a new operation."""
        self._reset_peak()
        self._t0 = time.perf_counter()
        self._start_alloc, _, _ = self._gpu_mem_mb()
        self._start_rss, _ = self._ram_mb()

    def _stop_timer(self):
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

    def record(self, model_key, quantization, phase, extra=None):
        """Stop timer and store a record with metadata."""
        metrics = self._stop_timer()
        metrics.update({"model": model_key, "quantization": quantization, "phase": phase})
        if extra:
            metrics.update(extra)
        self.records.append(metrics)
        self._print_record(metrics)
        return metrics

    def _print_record(self, m):
        model_sz = m.get('model_size_mb', 0)
        model_info = f"  model={model_sz:.1f}MB" if model_sz else ""
        print(f"  [{m['phase']}] {m['model']}@{m['quantization']}  "
              f"time={m['elapsed_s']}s  "
              f"VRAM peak={m['vram_peak_mb']}MB / {m['vram_total_mb']}MB  "
              f"RAM={m['ram_process_mb']}MB{model_info}")

    def _compute_averages(self):
        """Compute average metrics grouped by (model, quantization, phase)."""
        from collections import defaultdict

        numeric_keys = [
            "elapsed_s", "vram_allocated_mb", "vram_reserved_mb",
            "vram_peak_mb", "ram_process_mb", "ram_system_used_mb", "ram_delta_mb",
            "model_size_mb",
        ]
        groups = defaultdict(list)
        for r in self.records:
            key = (r["model"], r["quantization"], r["phase"])
            groups[key].append(r)

        averages = []
        for (model, quant, phase), records in groups.items():
            n = len(records)
            avg = {"model": model, "quantization": quant, "phase": phase, "count": n}
            for k in numeric_keys:
                vals = [r[k] for r in records if k in r]
                avg[f"avg_{k}"] = round(sum(vals) / len(vals), 2) if vals else 0.0
                avg[f"max_{k}"] = round(max(vals), 2) if vals else 0.0
            averages.append(avg)
        return averages

    def save_csv(self, path):
        """Write all collected records and averages to CSV files."""
        if not self.records:
            return
        fieldnames = list(self.records[0].keys())
        for r in self.records:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
                    print(f"Warning: new field '{k}' added to CSV header.")
                    
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)
        print(f"\nResource metrics saved to {path}")

        # Save averages CSV
        averages = self._compute_averages()
        if averages:
            avg_path = path.replace(".csv", "_averages.csv")
            avg_fields = list(averages[0].keys())
            with open(avg_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=avg_fields)
                writer.writeheader()
                writer.writerows(averages)
            print(f"Average metrics saved to {avg_path}")

    def summary(self):
        """Print a summary table of all records plus averages per model/quantization."""
        if not self.records:
            return
        print("\n" + "=" * 90)
        print(f"{'Model':<10} {'Quant':<6} {'Phase':<12} {'Time(s)':<9} "
              f"{'VRAM Peak(MB)':<15} {'Model(MB)':<12} {'RAM(MB)':<10}")
        print("-" * 100)
        for m in self.records:
            msz = m.get('model_size_mb', 0)
            print(f"{m['model']:<10} {m['quantization']:<6} {m['phase']:<12} "
                  f"{m['elapsed_s']:<9} {m['vram_peak_mb']:<15} {msz:<12.1f} {m['ram_process_mb']:<10}")
        print("=" * 100)

        # Print averages
        averages = self._compute_averages()
        gen_avgs = [a for a in averages if a["phase"] == "generate"]
        if gen_avgs:
            print("\n" + "=" * 100)
            print("AVERAGES PER MODEL / QUANTIZATION (generation only)")
            print("-" * 115)
            print(f"{'Model':<10} {'Quant':<6} {'N':<5} "
                  f"{'Avg Time(s)':<13} {'Max Time(s)':<13} "
                  f"{'Model(MB)':<12} "
                  f"{'Avg VRAM Peak':<15} {'Max VRAM Peak':<15} "
                  f"{'Avg RAM(MB)':<12}")
            print("-" * 115)
            for a in gen_avgs:
                print(f"{a['model']:<10} {a['quantization']:<6} {a['count']:<5} "
                      f"{a['avg_elapsed_s']:<13} {a['max_elapsed_s']:<13} "
                      f"{a.get('avg_model_size_mb', 0):<12.1f} "
                      f"{a['avg_vram_peak_mb']:<15} {a['max_vram_peak_mb']:<15} "
                      f"{a['avg_ram_process_mb']:<12}")
            print("=" * 115)

        load_avgs = [a for a in averages if a["phase"] == "load_model"]
        if load_avgs:
            print(f"\n{'Model':<10} {'Quant':<6} {'Load Time(s)':<13} "
                  f"{'Model(MB)':<12} {'VRAM Peak(MB)':<15} {'RAM(MB)':<12}")
            print("-" * 75)
            for a in load_avgs:
                print(f"{a['model']:<10} {a['quantization']:<6} {a['avg_elapsed_s']:<13} "
                      f"{a.get('avg_model_size_mb', 0):<12.1f} "
                      f"{a['avg_vram_peak_mb']:<15} {a['avg_ram_process_mb']:<12}")
