"""
FlashAttention-2 Benchmarking Script
=====================================
Compares Triton FlashAttention-2 (forward + backward) against standard PyTorch
attention using triton.testing.do_bench.

Settings:
  - batch_size = 1, causal = True
  - Sweep: seq_len × d × dtype
  - Reports forward, backward, and end-to-end latencies (ms)

Adapted for RTX 4060 (8 GB VRAM): max seq_len = 8192 by default.
"""

import itertools
import math
import sys
import argparse

import torch
import triton
import pandas as pd

from cs336_systems.flashattention2 import FlashAttentionPytorch
from cs336_systems.fused_attention import FlashAttentionTriton


def _print(*args, **kwargs):
    """Print with flush."""
    print(*args, **kwargs, flush=True)


# ── Vanilla PyTorch Attention (no FlashAttention backend) ─────────────────────

def pytorch_attention_forward(Q, K, V, is_causal=True):
    """Standard scaled dot-product attention using PyTorch ops only."""
    d = Q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    S = torch.bmm(Q, K.transpose(-2, -1)) * scale

    if is_causal:
        N_q, N_k = Q.shape[1], K.shape[1]
        mask = torch.triu(
            torch.ones(N_q, N_k, device=Q.device, dtype=torch.bool), diagonal=1
        )
        S = S.masked_fill(mask.unsqueeze(0), float("-inf"))

    P = torch.softmax(S, dim=-1)
    O = torch.bmm(P, V)
    return O


# ── Benchmark helpers ─────────────────────────────────────────────────────────

def make_qkv(batch, seq_len, d, dtype, device="cuda"):
    """Create random Q, K, V tensors requiring grad."""
    Q = torch.randn(batch, seq_len, d, device=device, dtype=dtype, requires_grad=True)
    K = torch.randn(batch, seq_len, d, device=device, dtype=dtype, requires_grad=True)
    V = torch.randn(batch, seq_len, d, device=device, dtype=dtype, requires_grad=True)
    return Q, K, V


def bench_forward(fn, Q, K, V, **kwargs):
    """Benchmark forward pass only."""
    def _fn():
        return fn(Q, K, V, **kwargs)
    ms = triton.testing.do_bench(_fn, warmup=25, rep=50)
    return ms


def bench_backward(fn, Q, K, V, **kwargs):
    """Benchmark backward pass only (forward is run to create the graph)."""
    O = fn(Q, K, V, **kwargs)
    dO = torch.randn_like(O)

    def _bwd():
        if Q.grad is not None:
            Q.grad = None
        if K.grad is not None:
            K.grad = None
        if V.grad is not None:
            V.grad = None
        O.backward(dO, retain_graph=True)

    ms = triton.testing.do_bench(_bwd, warmup=25, rep=50)
    return ms


def bench_e2e(fn, Q_orig, K_orig, V_orig, **kwargs):
    """Benchmark end-to-end forward + backward."""
    def _fn():
        Q = Q_orig.detach().clone().requires_grad_(True)
        K = K_orig.detach().clone().requires_grad_(True)
        V = V_orig.detach().clone().requires_grad_(True)
        O = fn(Q, K, V, **kwargs)
        dO = torch.ones_like(O)
        O.backward(dO)

    ms = triton.testing.do_bench(_fn, warmup=25, rep=50)
    return ms


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="FlashAttention-2 Benchmarking")
    p.add_argument(
        "--seq-lens", type=int, nargs="+",
        default=[128, 256, 512, 1024, 2048, 4096, 8192],
        help="Sequence lengths to benchmark",
    )
    p.add_argument(
        "--dims", type=int, nargs="+",
        default=[16, 32, 64, 128],
        help="Embedding dimensions to benchmark",
    )
    p.add_argument(
        "--dtypes", type=str, nargs="+",
        default=["bfloat16", "float32"],
        help="Data types (bfloat16, float32)",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--csv", type=str, default=None, help="Save results to CSV file")
    return p.parse_args()


def main():
    args = parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float16": torch.float16,
    }

    seq_lens = args.seq_lens
    dims = args.dims
    dtypes = [dtype_map[d] for d in args.dtypes]
    batch = args.batch_size

    rows = []
    configs = list(itertools.product(seq_lens, dims, dtypes))
    total = len(configs)

    _print(f"Running {total} configurations (batch={batch}, causal=True)")
    _print("=" * 130)

    for idx, (seq_len, d, dtype) in enumerate(configs, 1):
        dtype_name = str(dtype).split(".")[-1]
        tag = f"[{idx:3d}/{total}] seq={seq_len:5d}  d={d:3d}  dtype={dtype_name:10s}"
        _print(f"{tag}  ... starting")

        row = {
            "seq_len": seq_len,
            "d": d,
            "dtype": dtype_name,
        }

        # ── PyTorch vanilla attention ──
        try:
            Q, K, V = make_qkv(batch, seq_len, d, dtype)
            _print(f"  PT forward ...", end="")
            pt_fwd = bench_forward(pytorch_attention_forward, Q, K, V, is_causal=True)
            _print(f" {pt_fwd:.4f} ms", end="")

            _print(f"  |  bwd ...", end="")
            pt_bwd = bench_backward(pytorch_attention_forward, Q, K, V, is_causal=True)
            _print(f" {pt_bwd:.4f} ms", end="")

            _print(f"  |  e2e ...", end="")
            pt_e2e = bench_e2e(pytorch_attention_forward, Q, K, V, is_causal=True)
            _print(f" {pt_e2e:.4f} ms")

            row["pytorch_fwd_ms"] = f"{pt_fwd:.4f}"
            row["pytorch_bwd_ms"] = f"{pt_bwd:.4f}"
            row["pytorch_e2e_ms"] = f"{pt_e2e:.4f}"
            del Q, K, V
            torch.cuda.empty_cache()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            row["pytorch_fwd_ms"] = "OOM"
            row["pytorch_bwd_ms"] = "OOM"
            row["pytorch_e2e_ms"] = "OOM"
            torch.cuda.empty_cache()
            _print(f"  PyTorch: OOM ({e})")

        # ── Triton FlashAttention-2 ──
        # Wrap in lambda because autograd.Function.apply doesn't accept kwargs
        triton_fn = lambda q, k, v, is_causal=True: FlashAttentionTriton.apply(q, k, v, is_causal)
        try:
            Q, K, V = make_qkv(batch, seq_len, d, dtype)
            _print(f"  TR forward ...", end="")
            triton_fwd = bench_forward(triton_fn, Q, K, V, is_causal=True)
            _print(f" {triton_fwd:.4f} ms", end="")

            _print(f"  |  bwd ...", end="")
            triton_bwd = bench_backward(triton_fn, Q, K, V, is_causal=True)
            _print(f" {triton_bwd:.4f} ms", end="")

            _print(f"  |  e2e ...", end="")
            triton_e2e = bench_e2e(triton_fn, Q, K, V, is_causal=True)
            _print(f" {triton_e2e:.4f} ms")

            row["triton_fwd_ms"] = f"{triton_fwd:.4f}"
            row["triton_bwd_ms"] = f"{triton_bwd:.4f}"
            row["triton_e2e_ms"] = f"{triton_e2e:.4f}"
            del Q, K, V
            torch.cuda.empty_cache()
        except (torch.cuda.OutOfMemoryError, RuntimeError, Exception) as e:
            row["triton_fwd_ms"] = "OOM"
            row["triton_bwd_ms"] = "OOM"
            row["triton_e2e_ms"] = "OOM"
            torch.cuda.empty_cache()
            _print(f"  Triton: Error ({type(e).__name__}: {e})")

        rows.append(row)

    # ── Print result table ────────────────────────────────────────────────────
    _print("\n" + "=" * 130)
    _print("RESULTS TABLE")
    _print("=" * 130)

    df = pd.DataFrame(rows)
    _print(df.to_string(index=False))

    if args.csv:
        df.to_csv(args.csv, index=False)
        _print(f"\nResults saved to {args.csv}")


if __name__ == "__main__":
    main()
