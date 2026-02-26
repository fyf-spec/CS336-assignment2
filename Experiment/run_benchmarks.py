"""Run benchmarking_script.py for all 5 model sizes and output a LaTeX table."""

import subprocess
import sys
import re
from pathlib import Path
import pandas as pd

# ── Table 1: Specifications of different model sizes ──────────────────────────
MODEL_CONFIGS = [
    {"size": "small",  "d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    {"size": "medium", "d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    {"size": "large",  "d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    {"size": "xl",     "d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
    {"size": "2.7B",   "d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
]


def run_benchmark(config: dict) -> dict:
    """Run benchmarking_script.py with the given config and return parsed timing results."""
    cmd = [
        sys.executable, str(Path(__file__).parent / "benchmarking_script.py"),
        "--d_model", str(config["d_model"]),
        "--d_ff", str(config["d_ff"]),
        "--num-layers", str(config["num_layers"]),
        "--num-heads", str(config["num_heads"]),
        "--warm_up", "5",
        "--num-runs", "10",
    ]

    print(f"\n{'='*60}")
    print(f"  Running: {config['size']}  "
          f"(d_model={config['d_model']}, d_ff={config['d_ff']}, "
          f"num_layers={config['num_layers']}, num_heads={config['num_heads']})")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print subprocess output for visibility
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(f"  ⚠ Benchmark for '{config['size']}' failed (exit code {result.returncode})")
        return {"avg_fwd": float("nan"), "avg_bwd": float("nan"), "avg_total": float("nan")}

    # Parse the three timing lines from stdout
    fwd = _parse_line(result.stdout, r"Average forward pass time:\s*([\d.eE+-]+)")
    bwd = _parse_line(result.stdout, r"Average backward pass time:\s*([\d.eE+-]+)")
    total = _parse_line(result.stdout, r"Average total time:\s*([\d.eE+-]+)")

    return {"avg_fwd": fwd, "avg_bwd": bwd, "avg_total": total}


def _parse_line(text: str, pattern: str) -> float:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else float("nan")


def results_to_latex(configs: list[dict], results: list[dict]) -> str:
    """Build a DataFrame from configs + results and return a LaTeX table string."""
    rows = []
    for cfg, res in zip(configs, results):
        rows.append({
            "Size": cfg["size"],
            "d_model": cfg["d_model"],
            "d_ff": cfg["d_ff"],
            "num_layers": cfg["num_layers"],
            "num_heads": cfg["num_heads"],
            "Avg Forward (s)": f"{res['avg_fwd']:.4f}",
            "Avg Backward (s)": f"{res['avg_bwd']:.4f}",
            "Avg Total (s)": f"{res['avg_total']:.4f}",
        })

    df = pd.DataFrame(rows)
    latex = df.to_latex(index=False, caption="Benchmarking results for different model sizes", label="tab:benchmark")
    return latex


def main():
    results = []
    for cfg in MODEL_CONFIGS:
        res = run_benchmark(cfg)
        results.append(res)

    latex_table = results_to_latex(MODEL_CONFIGS, results)

    print("\n" + "=" * 60)
    print("  LaTeX Table Output")
    print("=" * 60)
    print(latex_table)

    # Also save to file
    output_path = "benchmark_results.tex"
    with open(output_path, "w") as f:
        f.write(latex_table)
    print(f"LaTeX table saved to {output_path}")


if __name__ == "__main__":
    main()
