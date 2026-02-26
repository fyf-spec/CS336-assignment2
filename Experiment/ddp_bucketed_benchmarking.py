"""DDP Bucketed Benchmarking — varying bucket sizes.

Compares bucketed DDP with different max bucket sizes (1, 10, 100, 1000 MB)
plus the baseline methods (naive per-param, flat single call, overlapping per-param).

Uses mp.spawn + gloo pattern (same as tests).

Usage:
    CUDA_VISIBLE_DEVICES=3,4 uv run python Experiment/ddp_bucketed_benchmarking.py
    CUDA_VISIBLE_DEVICES=3,4 uv run python Experiment/ddp_bucketed_benchmarking.py --mixed-precision
"""

import os
import time
import argparse
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_systems.naive_ddp import DDPIndividualParameters, DDPBucketed

# ── Process group helpers ────────────────────────────────────────────────────
def _setup_process_group(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12390"
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > 0:
            local_rank = rank % device_count
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
        else:
            raise ValueError("No CUDA devices found.")
    else:
        device = "cpu"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device

def _cleanup_process_group():
    dist.barrier()
    dist.destroy_process_group()

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-size", default="small", choices=list(MODEL_CONFIGS.keys()))
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--warmup-iters", type=int, default=3)
    p.add_argument("--bench-iters", type=int, default=10)
    p.add_argument("--mixed-precision", action="store_true", default=False)
    return p.parse_args()


def bench_method(ddp_model, optimizer, scaler, autocast_ctx, args, device, rank, label):
    """Warmup + benchmark. Communication happens via hooks during backward.
    finish_gradient_synchronization() waits on handles."""
    for _ in range(args.warmup_iters):
        x = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)
        y = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            logits = ddp_model(x)
            loss = cross_entropy(logits, y)
        scaler.scale(loss).backward()
        ddp_model.finish_gradient_synchronization()
        scaler.step(optimizer)
        scaler.update()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    total_ms, wait_ms = [], []
    for it in range(args.bench_iters):
        x = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)
        y = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            logits = ddp_model(x)
            loss = cross_entropy(logits, y)
        scaler.scale(loss).backward()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tw0 = time.perf_counter()
        ddp_model.finish_gradient_synchronization()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tw1 = time.perf_counter()

        scaler.step(optimizer)
        scaler.update()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        total_ms.append((t1 - t0) * 1000)
        wait_ms.append((tw1 - tw0) * 1000)
        if rank == 0:
            print(f"  [{label:>16}] iter {it+1:2d}: total={total_ms[-1]:.1f}ms  "
                  f"wait={wait_ms[-1]:.1f}ms  wait%={wait_ms[-1]/total_ms[-1]*100:.1f}%", flush=True)

    return sum(total_ms) / len(total_ms), sum(wait_ms) / len(wait_ms)


def bench_non_overlapping(ddp_model, optimizer, scaler, autocast_ctx, args, device, rank, label, sync_fn):
    """Benchmark for naive/flat methods (no hooks, sync after backward)."""
    for _ in range(args.warmup_iters):
        x = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)
        y = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            logits = ddp_model(x)
            loss = cross_entropy(logits, y)
        scaler.scale(loss).backward()
        sync_fn()
        scaler.step(optimizer)
        scaler.update()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    total_ms, comm_ms = [], []
    for it in range(args.bench_iters):
        x = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)
        y = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            logits = ddp_model(x)
            loss = cross_entropy(logits, y)
        scaler.scale(loss).backward()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tc0 = time.perf_counter()
        sync_fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tc1 = time.perf_counter()

        scaler.step(optimizer)
        scaler.update()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        total_ms.append((t1 - t0) * 1000)
        comm_ms.append((tc1 - tc0) * 1000)
        if rank == 0:
            print(f"  [{label:>16}] iter {it+1:2d}: total={total_ms[-1]:.1f}ms  "
                  f"comm={comm_ms[-1]:.1f}ms  comm%={comm_ms[-1]/total_ms[-1]*100:.1f}%", flush=True)

    return sum(total_ms) / len(total_ms), sum(comm_ms) / len(comm_ms)


# ── Worker ────────────────────────────────────────────────────────────────────
def _worker(rank, world_size, args):
    device = _setup_process_group(rank, world_size, backend="gloo")
    dist.barrier()

    cfg = MODEL_CONFIGS[args.model_size]
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"  DDP Bucketed Benchmark | model={args.model_size} | world_size={world_size}")
        print(f"  bs/rank={args.batch_size} | seq={args.seq_len} | mp={args.mixed_precision}")
        print(f"  device={device} | backend=gloo")
        print(f"{'='*70}", flush=True)

    torch.manual_seed(42)
    base_model = TransformerLM(
        vocab_size=args.vocab_size, context_length=args.seq_len,
        d_model=cfg["d_model"], num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"], d_ff=cfg["d_ff"],
        rope_theta=10000.0, device=device,
    )
    n_params = sum(p.numel() for p in base_model.parameters())
    n_tensors = len(list(base_model.parameters()))
    param_mb = n_params * 4 / (1024 * 1024)  # assuming float32
    if rank == 0:
        print(f"  params={n_params:,} ({n_tensors} tensors, ~{param_mb:.1f} MB)\n", flush=True)

    autocast_ctx = torch.autocast("cuda", dtype=torch.float16, enabled=args.mixed_precision)
    results = []

    # ── Baseline: naive per-parameter (no hooks) ──
    if rank == 0:
        print(f"── naive: sync per-parameter ──", flush=True)
    m = DDPIndividualParameters(deepcopy(base_model), use_hooks=False)
    o = torch.optim.AdamW(m.parameters(), lr=1e-4)
    s = torch.amp.GradScaler("cuda", enabled=args.mixed_precision)
    t, c = bench_non_overlapping(m, o, s, autocast_ctx, args, device, rank,
                                 "naive", m.finish_gradient_synchronization_naive)
    results.append(("naive", t, c))

    # ── Baseline: flat (no hooks) ──
    if rank == 0:
        print(f"\n── flat: single all-reduce ──", flush=True)
    m = DDPIndividualParameters(deepcopy(base_model), use_hooks=False)
    o = torch.optim.AdamW(m.parameters(), lr=1e-4)
    s = torch.amp.GradScaler("cuda", enabled=args.mixed_precision)
    t, c = bench_non_overlapping(m, o, s, autocast_ctx, args, device, rank,
                                 "flat", m.finish_gradient_synchronization_flat)
    results.append(("flat", t, c))

    # ── Baseline: overlapping per-param (hooks) ──
    if rank == 0:
        print(f"\n── overlap-individual: async per-parameter ──", flush=True)
    m = DDPIndividualParameters(deepcopy(base_model), use_hooks=True)
    o = torch.optim.AdamW(m.parameters(), lr=1e-4)
    s = torch.amp.GradScaler("cuda", enabled=args.mixed_precision)
    t, c = bench_method(m, o, s, autocast_ctx, args, device, rank, "overlap-indiv")
    results.append(("overlap-indiv", t, c))

    # ── Bucketed: varying bucket sizes ──
    bucket_sizes_mb = [1, 10, 100, 1000]
    for bsz in bucket_sizes_mb:
        label = f"bucket-{bsz}MB"
        if rank == 0:
            print(f"\n── {label} ──", flush=True)
        m = DDPBucketed(deepcopy(base_model), bucket_size_mb=bsz)
        n_buckets = len(m._buckets)
        if rank == 0:
            print(f"  ({n_buckets} buckets)", flush=True)
        o = torch.optim.AdamW(m.parameters(), lr=1e-4)
        s = torch.amp.GradScaler("cuda", enabled=args.mixed_precision)
        t, c = bench_method(m, o, s, autocast_ctx, args, device, rank, label)
        results.append((label, t, c))

    # ── Summary table ──
    if rank == 0:
        naive_total = results[0][1]
        print(f"\n{'='*70}")
        print(f"  RESULTS ({args.model_size}, world_size={world_size})")
        print(f"{'='*70}")
        print(f"  {'Method':<18} {'Total ms':>10} {'Comm/Wait ms':>14} {'Overhead%':>10} {'vs naive':>10}")
        print(f"  {'─'*18} {'─'*10} {'─'*14} {'─'*10} {'─'*10}")
        for name, t, c in results:
            speedup = naive_total / t
            print(f"  {name:<18} {t:>10.1f} {c:>14.1f} {c/t*100:>9.1f}% {speedup:>9.2f}x")
        print(f"{'='*70}", flush=True)

    _cleanup_process_group()


def main():
    args = parse_args()
    world_size = 2
    mp.spawn(_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
