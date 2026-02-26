"""Memory profiling script for the 2.7B Transformer model.

Supports forward-only (inference) and full training step modes,
with optional mixed-precision. Outputs a .pickle snapshot for
https://pytorch.org/memory_viz .

Usage examples:
    # Forward-only, context length 128
    python Experiment/memory_profiling.py --context-length 128 --mode forward

    # Full training step, context length 512, mixed precision
    python Experiment/memory_profiling.py --context-length 512 --mode train --mixed-precision

    # Custom output filename
    python Experiment/memory_profiling.py --context-length 256 --mode train -o my_snapshot.pickle
"""

import argparse
import torch
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy

# ── 2.7B model config from Table 1 ───────────────────────────────────────────
MODEL_CONFIG = {
    "d_model": 2560,
    "d_ff": 10240,
    "num_layers": 32,
    "num_heads": 32,
}

VOCAB_SIZE = 10000
THETA = 10000.0


def parse_args():
    parser = argparse.ArgumentParser(description="Memory profiling for 2.7B model")
    parser.add_argument("--context-length", type=int, required=True,
                        choices=[128, 256, 512], help="Context length to profile")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["forward", "train"],
                        help="'forward' for inference only, 'train' for full training step")
    parser.add_argument("--mixed-precision", action="store_true", default=False,
                        help="Enable mixed precision (fp16)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output pickle filename (auto-generated if not specified)")
    parser.add_argument("--warm-up", type=int, default=2, help="Number of warmup iterations")
    return parser.parse_args()


def create_batch(batch_size, seq_len, vocab_size, device):
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return x, y


def run_forward(model, x, y, autocast_ctx):
    """Run forward pass only."""
    with autocast_ctx:
        logits = model(x)
        loss = cross_entropy(logits, y)
    return loss


def run_train_step(model, x, y, optimizer, autocast_ctx, scaler):
    """Run a full training step: forward + backward + optimizer."""
    optimizer.zero_grad(set_to_none=True)
    with autocast_ctx:
        logits = model(x)
        loss = cross_entropy(logits, y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss


def main():
    args = parse_args()
    device = "cuda"

    # Auto-generate output filename
    if args.output is None:
        mp_tag = "_mp" if args.mixed_precision else ""
        args.output = f"memory_snapshot_{args.mode}_ctx{args.context_length}{mp_tag}.pickle"

    print(f"Config: 2.7B model, context_length={args.context_length}, "
          f"mode={args.mode}, mixed_precision={args.mixed_precision}, "
          f"batch_size={args.batch_size}")
    print(f"Output: {args.output}")

    # ── Build model ───────────────────────────────────────────────────────────
    print("Building model...")
    model = TransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=args.context_length,
        d_model=MODEL_CONFIG["d_model"],
        num_layers=MODEL_CONFIG["num_layers"],
        num_heads=MODEL_CONFIG["num_heads"],
        d_ff=MODEL_CONFIG["d_ff"],
        rope_theta=THETA,
        device=device,
    ).to(device)

    # ── Optimizer (only needed for train mode) ────────────────────────────────
    optimizer = None
    if args.mode == "train":
        optimizer = AdamW(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )

    # ── Mixed precision setup ─────────────────────────────────────────────────
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16,
                                  enabled=args.mixed_precision)
    scaler = torch.amp.GradScaler("cuda", enabled=args.mixed_precision)

    # ── Warmup ────────────────────────────────────────────────────────────────
    print(f"Warming up ({args.warm_up} iters)...")
    for i in range(args.warm_up):
        x, y = create_batch(args.batch_size, args.context_length, VOCAB_SIZE, device)
        if args.mode == "forward":
            with torch.no_grad():
                run_forward(model, x, y, autocast_ctx)
        else:
            run_train_step(model, x, y, optimizer, autocast_ctx, scaler)
        torch.cuda.synchronize()
        print(f"  warmup {i+1}/{args.warm_up} done")

    # ── Clear memory and start recording ──────────────────────────────────────
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    print("Starting memory recording...")
    torch.cuda.memory._record_memory_history(max_entries=1000000)

    # ── Profiled iteration ────────────────────────────────────────────────────
    x, y = create_batch(args.batch_size, args.context_length, VOCAB_SIZE, device)

    if args.mode == "forward":
        with torch.no_grad():
            run_forward(model, x, y, autocast_ctx)
    else:
        run_train_step(model, x, y, optimizer, autocast_ctx, scaler)

    torch.cuda.synchronize()

    # ── Dump snapshot and stop recording ───────────────────────────────────────
    print(f"Saving memory snapshot to {args.output}...")
    torch.cuda.memory._dump_snapshot(args.output)
    torch.cuda.memory._record_memory_history(enabled=None)

    # ── Report peak memory ────────────────────────────────────────────────────
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    current_mem = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"\nPeak memory allocated:    {peak_mem:.1f} MiB")
    print(f"Current memory allocated: {current_mem:.1f} MiB")
    print(f"\nDone! Load {args.output} at https://pytorch.org/memory_viz")


if __name__ == "__main__":
    main()
