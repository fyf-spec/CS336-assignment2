import cs336_basics.nvtx_wrapper  # activate NVTX annotations (must be before from-imports)
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW, get_cosine_lr
from cs336_basics.nn_utils import cross_entropy
import timeit
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmarking script for basic model")
    # data loader arguments
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size")
    
    # transformer arguments
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--num-heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=3072, help="Feed-forward dimension")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--theta", type=float, default=10000.0, help="Theta parameter for RoPE")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of transformer layers")

    # optimizer arguments
    parser.add_argument("--cosine_cycle_iters", type=int, default=60000)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--betas", type=tuple[float, float], default=(0.9, 0.999), help="Beta parameters for AdamW")
    parser.add_argument("--eps", type=float, default=1e-8, help="Epsilon parameter for AdamW")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for AdamW")
    
    # training loop arguments
    parser.add_argument("--warm_up", type=int, default=5, help="Number of warm up runs")
    parser.add_argument("--num-runs", type=int, default=5, help="Number of runs")
    
    # other settings
    parser.add_argument("--device", type=str, default='cuda', help="Device to run the model on")
    parser.add_argument("--mixed-precision", action="store_true", default=False, help="Enable mixed precision training (fp16)")
    return parser.parse_args()

def create_batch(
    batch_size: int = 4,
    seq_len: int = 128,
    vocab_size: int = 10000,
):
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    y = torch.randint(0, vocab_size, (batch_size, seq_len))
    return x, y

def main():
    args = parse_args()

    # set transformer model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.max_seq_len,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.theta,
        device='cuda'
    ).to(args.device)

    # set optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    # mixed precision setup
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.mixed_precision)
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

    torch.cuda.synchronize()
    time_fwd = 0.0
    time_bwd = 0.0

    print("running for {} iterations".format(args.num_runs +args.warm_up))
    print("warm_up starts")
    for it in range(1, args.warm_up + args.num_runs+1):
        if it == args.warm_up + 1: print("warmup completed, starting to record")
        # get lr
        lr = get_cosine_lr(
            it, 
            max_learning_rate=args.lr, 
            min_learning_rate=args.min_lr, 
            warmup_iters=args.warm_up, 
            cosine_cycle_iters=args.cosine_cycle_iters
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        x, y = create_batch(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
        )
        x, y = x.to(args.device), y.to(args.device)
    
        # forward pass
        torch.cuda.synchronize()
        start_fwd = timeit.default_timer()
        with autocast_ctx:
            logits = model(x)
            loss = cross_entropy(logits, y)
        torch.cuda.synchronize()
        end_fwd = timeit.default_timer()
        if it > args.warm_up: time_fwd += end_fwd - start_fwd

        # backward pass
        torch.cuda.synchronize()
        start_bwd = timeit.default_timer()
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        end_bwd = timeit.default_timer()
        if it > args.warm_up: time_bwd += end_bwd - start_bwd

        print("iteration {}/{} completed".format(it, args.warm_up + args.num_runs))

    print(f"Average forward pass time: {time_fwd / args.num_runs}")
    print(f"Average backward pass time: {time_bwd / args.num_runs}")
    print(f"Average total time: {(time_fwd + time_bwd) / args.num_runs}")

if __name__ == "__main__":
    main()