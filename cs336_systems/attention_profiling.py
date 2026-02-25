"""Benchmark attention at different scales and output a LaTeX table.

Iterates through the cartesian product of:
  d_model ∈ [16, 32, 64, 128]
  seq_len ∈ [256, 1024, 4096, 8192, 16384]

Fixed: batch_size = 8, no multi-head (single head).
Measures: forward time, memory before backward, backward time (100 runs each).
"""

import itertools
import timeit
import torch
import pandas as pd
import torch.nn.functional as F
import math


BATCH_SIZE = 8
D_MODELS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384]
NUM_WARMUP = 5
NUM_RUNS = 100

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark attention at different scales")
    parse.add_aegument("--compile", type=bool, default=False, help="Compile model with torch.compile")
    return parser.parse_args()

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    """Causal scaled dot-product attention.

    Args:
        Q: (batch, seq_len, d_model)
        K: (batch, seq_len, d_model)
        V: (batch, seq_len, d_model)

    Returns:
        Output tensor of shape (batch, seq_len, d_model).
    """
    d_k = Q.shape[-1]
    # (batch, seq_len, seq_len)
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)

    # Causal mask: upper triangle = -inf
    seq_len = Q.shape[1]
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool),
        diagonal=1,
    )
    scores = scores.masked_fill(causal_mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, V)

def benchmark_attention(d_model: int, seq_len: int, device: str = "cuda") -> dict:
    """Benchmark a single (d_model, seq_len) configuration."""

    attention = scaled_dot_product_attention
    if args.compile:
        attention = torch.compile(attention)

    Q = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, requires_grad=True)
    K = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, requires_grad=True)
    V = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, requires_grad=True)

    # ── Warm up ───────────────────────────────────────────────────────────────
    for _ in range(NUM_WARMUP):
        out = attention(Q, K, V)
        loss = out.sum()
        loss.backward()
        Q.grad = None; K.grad = None; V.grad = None
        torch.cuda.synchronize()

    # ── Forward pass timing ───────────────────────────────────────────────────
    torch.cuda.synchronize()
    time_fwd = 0.0
    for _ in range(NUM_RUNS):
        torch.cuda.synchronize()
        start = timeit.default_timer()
        out = attention(Q, K, V)
        torch.cuda.synchronize()
        time_fwd += timeit.default_timer() - start

    # ── Memory measurement (before backward) ─────────────────────────────────
    torch.cuda.synchronize()
    mem_before_bwd = torch.cuda.memory_allocated() / (1024 ** 2)  # MiB

    # ── Backward pass timing ─────────────────────────────────────────────────
    time_bwd = 0.0
    for _ in range(NUM_RUNS):
        # Re-run forward to get a fresh computation graph
        out = attention(Q, K, V)
        loss = out.sum()
        Q.grad = None; K.grad = None; V.grad = None
        torch.cuda.synchronize()
        start = timeit.default_timer()
        loss.backward()
        torch.cuda.synchronize()
        time_bwd += timeit.default_timer() - start

    avg_fwd = time_fwd / NUM_RUNS
    avg_bwd = time_bwd / NUM_RUNS

    return {
        "avg_fwd": avg_fwd,
        "avg_bwd": avg_bwd,
        "mem_MiB": mem_before_bwd,
    }


def main():
    rows = []
    configs = list(itertools.product(D_MODELS, SEQ_LENS))

    for i, (d_model, seq_len) in enumerate(configs, 1):
        label = f"[{i}/{len(configs)}] d_model={d_model}, seq_len={seq_len}"
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        try:
            # Clear GPU memory from previous run
            torch.cuda.empty_cache()
            res = benchmark_attention(d_model, seq_len)
            print(f"  Avg forward:  {res['avg_fwd']:.6f} s")
            print(f"  Avg backward: {res['avg_bwd']:.6f} s")
            print(f"  Memory:       {res['mem_MiB']:.1f} MiB")
        except torch.cuda.OutOfMemoryError:
            print("  ⚠ OOM — skipping this configuration")
            res = {"avg_fwd": float("nan"), "avg_bwd": float("nan"), "mem_MiB": float("nan")}

        rows.append({
            "d_model": d_model,
            "seq_len": seq_len,
            "Avg Forward (s)": res["avg_fwd"],
            "Avg Backward (s)": res["avg_bwd"],
            "Memory (MiB)": res["mem_MiB"],
        })

    # ── Build DataFrame and output LaTeX ──────────────────────────────────────
    df = pd.DataFrame(rows)

    # Format numeric columns
    for col in ["Avg Forward (s)", "Avg Backward (s)"]:
        df[col] = df[col].apply(lambda x: f"{x:.6f}" if not pd.isna(x) else "OOM")
    df["Memory (MiB)"] = df["Memory (MiB)"].apply(lambda x: f"{x:.1f}" if not pd.isna(x) else "OOM")

    latex = df.to_latex(
        index=False,
        caption="Attention benchmark results at different scales (batch\\_size=8, single head)",
        label="tab:attention_benchmark",
    )

    print("\n" + "=" * 60)
    print("  LaTeX Table Output")
    print("=" * 60)
    print(latex)

    output_path = "attention_benchmark_results.tex"
    with open(output_path, "w") as f:
        f.write(latex)
    print(f"LaTeX table saved to {output_path}")


if __name__ == "__main__":
    main()
