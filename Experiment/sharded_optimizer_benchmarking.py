"""Optimizer State Sharding Benchmark
=====================================
Profiles peak memory and training speed with/without optimizer state sharding.

Configuration: 1 node, 2 GPUs, XL-equivalent model.

Usage:
    CUDA_VISIBLE_DEVICES=3,4 uv run python Experiment/sharded_optimizer_benchmarking.py
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
from cs336_systems.naive_ddp import DDPBucketed
from cs336_systems.sharded_optimizer import ShardedOptimizer

# ── Process group helpers ────────────────────────────────────────────────────
def _setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12390"
    if torch.cuda.is_available():
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device

def _cleanup():
    dist.barrier()
    dist.destroy_process_group()

# ── Config ────────────────────────────────────────────────────────────────────
# "small" model to fit on GPUs. Adjust if OOM.
MODEL_CONFIGS = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-size", default="medium", choices=list(MODEL_CONFIGS.keys()))
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--warmup-iters", type=int, default=2)
    p.add_argument("--bench-iters", type=int, default=5)
    p.add_argument("--mixed-precision", action="store_true", default=False)
    p.add_argument("--bucket-size-mb", type=float, default=100.0)
    return p.parse_args()


def fmt_mb(b):
    return f"{b / 1024**2:.1f} MB"


def get_cuda_mem():
    """Return dict of current CUDA memory stats in bytes."""
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "peak_allocated": torch.cuda.max_memory_allocated(),
    }


def profile_one_setting(base_model, args, device, rank, use_sharding: bool):
    """Profile memory + speed for a single setting."""
    label = "SHARDED" if use_sharding else "NON-SHARDED"
    if rank == 0:
        print(f"\n{'─'*60}")
        print(f"  {label} optimizer")
        print(f"{'─'*60}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # ── 1. Model init ─────────────────────────────────────────────────────
    ddp_model = DDPBucketed(base_model, bucket_size_mb=args.bucket_size_mb)

    if rank == 0:
        mem_after_model = get_cuda_mem()
        print(f"  After model init:       alloc={fmt_mb(mem_after_model['allocated']):>10}  "
              f"peak={fmt_mb(mem_after_model['peak_allocated']):>10}", flush=True)

    # ── 2. Optimizer init ─────────────────────────────────────────────────
    if use_sharding:
        optimizer = ShardedOptimizer(
            ddp_model.parameters(), torch.optim.AdamW,
            lr=1e-4, weight_decay=0.01
        )
    else:
        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4, weight_decay=0.01)

    if rank == 0:
        mem_after_optim = get_cuda_mem()
        print(f"  After optimizer init:   alloc={fmt_mb(mem_after_optim['allocated']):>10}  "
              f"peak={fmt_mb(mem_after_optim['peak_allocated']):>10}", flush=True)

    autocast_ctx = torch.autocast("cuda", dtype=torch.float16, enabled=args.mixed_precision)
    scaler = torch.amp.GradScaler("cuda", enabled=args.mixed_precision)

    # ── 3. Warmup ─────────────────────────────────────────────────────────
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
        torch.cuda.synchronize()

    # ── 4. Benchmark iterations with memory snapshots ─────────────────────
    torch.cuda.reset_peak_memory_stats()

    total_times = []
    for it in range(args.bench_iters):
        x = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)
        y = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            logits = ddp_model(x)
            loss = cross_entropy(logits, y)
        scaler.scale(loss).backward()
        ddp_model.finish_gradient_synchronization()

        if rank == 0 and it == 0:
            torch.cuda.synchronize()
            mem_before_step = get_cuda_mem()
            print(f"  Before optimizer step:  alloc={fmt_mb(mem_before_step['allocated']):>10}  "
                  f"peak={fmt_mb(mem_before_step['peak_allocated']):>10}", flush=True)

        scaler.step(optimizer)
        scaler.update()

        if rank == 0 and it == 0:
            torch.cuda.synchronize()
            mem_after_step = get_cuda_mem()
            print(f"  After optimizer step:   alloc={fmt_mb(mem_after_step['allocated']):>10}  "
                  f"peak={fmt_mb(mem_after_step['peak_allocated']):>10}", flush=True)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        total_times.append((t1 - t0) * 1000)

        if rank == 0:
            print(f"    iter {it+1}: {total_times[-1]:.1f} ms", flush=True)

    avg_time = sum(total_times) / len(total_times)

    if rank == 0:
        peak = torch.cuda.max_memory_allocated()
        print(f"\n  Avg step time:  {avg_time:.1f} ms")
        print(f"  Overall peak allocated: {fmt_mb(peak)}", flush=True)

    # Clean up model/optimizer to free GPU memory for next setting
    del ddp_model, optimizer
    torch.cuda.empty_cache()
    dist.barrier()

    return avg_time if rank == 0 else 0.0


def _worker(rank, world_size, args, use_sharding):
    """Worker for a single setting (sharded or non-sharded)."""
    device = _setup(rank, world_size, backend="gloo")
    dist.barrier()

    cfg = MODEL_CONFIGS[args.model_size]
    label = "SHARDED" if use_sharding else "NON-SHARDED"

    torch.manual_seed(42)
    model = TransformerLM(
        vocab_size=args.vocab_size, context_length=args.seq_len,
        d_model=cfg["d_model"], num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"], d_ff=cfg["d_ff"],
        rope_theta=10000.0, device=device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    param_mb = n_params * 4 / (1024**2)

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  {label} | model={args.model_size} | ws={world_size}")
        print(f"  params={n_params:,} (~{param_mb:.1f} MB in fp32)")
        print(f"{'='*60}", flush=True)

    profile_one_setting(model, args, device, rank, use_sharding)

    _cleanup()


def main():
    import gc
    args = parse_args()

    cfg = MODEL_CONFIGS[args.model_size]
    n_params = sum(p.numel() for p in TransformerLM(
        vocab_size=args.vocab_size, context_length=args.seq_len,
        d_model=cfg["d_model"], num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"], d_ff=cfg["d_ff"],
        rope_theta=10000.0, device="cpu",
    ).parameters())
    param_mb = n_params * 4 / (1024**2)
    print(f"Model: {args.model_size}, params={n_params:,} (~{param_mb:.1f} MB fp32)")
    print(f"AdamW m+v: non-sharded={param_mb*2:.1f} MB, sharded(/2)={param_mb:.1f} MB")
    print(f"Expected savings: ~{param_mb:.1f} MB")

    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs")

    # Run each setting in a SEPARATE mp.spawn for clean GPU memory
    print("\n>>> Running NON-SHARDED <<<")
    mp.spawn(_worker, args=(world_size, args, False), nprocs=world_size, join=True)

    gc.collect()
    torch.cuda.empty_cache()

    print("\n>>> Running SHARDED <<<")
    mp.spawn(_worker, args=(world_size, args, True), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
