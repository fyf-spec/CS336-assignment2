"""Distributed Data Parallel (DDP) implementations.

Three gradient sync strategies:
  1. Per-parameter synchronous all-reduce (naive)
  2. Flat all-reduce (single call)
  3. Overlapping async all-reduce via post_accumulate_grad_hook (default)
"""

import torch
import torch.nn as nn
import torch.distributed as dist


class DDPIndividualParameters(nn.Module):
    """DDP wrapper that overlaps gradient communication with backward computation.

    On construction:
      - Broadcasts all parameters from rank 0 to all other ranks.
      - Registers post_accumulate_grad_hook on each unique parameter to
        kick off async all-reduce as soon as the gradient is ready.

    Call `finish_gradient_synchronization()` after backward but before
    optimizer.step() to wait for all async all-reduce calls to complete.
    """

    def __init__(self, module: nn.Module, use_hooks: bool = True):
        super().__init__()
        self.module = module
        self._handles: list[tuple[dist.Work, torch.nn.Parameter]] = []

        # Broadcast parameters from rank 0 to all ranks
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # Register hooks for async gradient sync (skip tied weights)
        if use_hooks:
            seen_ids = set()
            for param in self.module.parameters():
                if id(param) in seen_ids:
                    continue
                seen_ids.add(id(param))
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(self._make_hook())

    def _make_hook(self):
        """Create a hook that launches async all-reduce when grad is ready."""
        def hook(param: torch.Tensor):
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self._handles.append((handle, param))
        return hook

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        """Wait for all async all-reduce calls, then average gradients."""
        for handle, param in self._handles:
            handle.wait()
            param.grad /= dist.get_world_size()
        self._handles.clear()

    # ── Alternative sync methods (for benchmarking comparison) ────────────────

    def finish_gradient_synchronization_naive(self):
        """Synchronous per-parameter all-reduce (no overlap)."""
        seen_param_ids = set()
        for param in self.module.parameters():
            if id(param) in seen_param_ids:
                continue
            seen_param_ids.add(id(param))
            if param.requires_grad and param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= dist.get_world_size()

    def finish_gradient_synchronization_flat(self):
        """Flatten all grads into one tensor, single all-reduce, then unflatten."""
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

        seen_ids = set()
        grads = []
        params_with_grad = []
        for param in self.module.parameters():
            if id(param) in seen_ids:
                continue
            seen_ids.add(id(param))
            if param.requires_grad and param.grad is not None:
                grads.append(param.grad)
                params_with_grad.append(param)

        if not grads:
            return

        # Flatten → single all_reduce → unflatten
        flat = _flatten_dense_tensors(grads)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat /= dist.get_world_size()

        # Copy results back to individual grad tensors
        for param, unflat in zip(params_with_grad, _unflatten_dense_tensors(flat, grads)):
            param.grad.copy_(unflat)


class DDPBucketed(nn.Module):
    """DDP wrapper with gradient bucketing and overlapping communication.

    Parameters are assigned to buckets (in reverse order of model.parameters())
    up to `bucket_size_mb` megabytes each. During backward, a hook on each
    parameter tracks when its gradient is ready. When all gradients in a bucket
    are ready, an async all-reduce is launched for the flattened bucket.

    Call `finish_gradient_synchronization()` after backward to wait on all
    outstanding async handles and scatter averaged gradients back.
    """

    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self._handles: list[tuple[dist.Work, int]] = []  # (handle, bucket_idx)

        # Broadcast parameters from rank 0
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # ── Build buckets in reverse parameter order ──────────────────────────
        # Collect unique parameters (skip tied weights)
        seen_ids = set()
        unique_params = []
        for param in self.module.parameters():
            if id(param) in seen_ids:
                continue
            seen_ids.add(id(param))
            if param.requires_grad:
                unique_params.append(param)

        # Reverse order: gradients arrive approximately in reverse of
        # model.parameters() during backward
        reversed_params = list(reversed(unique_params))

        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        self._buckets: list[list[nn.Parameter]] = []  # bucket_idx -> [params]
        self._param_to_bucket: dict[int, int] = {}    # param id -> bucket_idx
        self._bucket_pending: list[int] = []           # how many grads still pending per bucket

        current_bucket: list[nn.Parameter] = []
        current_size = 0.0

        for param in reversed_params:
            param_bytes = param.numel() * param.element_size()
            # Start new bucket if adding this param exceeds limit
            # (unless current bucket is empty — always add at least one param)
            if current_bucket and current_size + param_bytes > bucket_size_bytes:
                bucket_idx = len(self._buckets)
                self._buckets.append(current_bucket)
                self._bucket_pending.append(len(current_bucket))
                for p in current_bucket:
                    self._param_to_bucket[id(p)] = bucket_idx
                current_bucket = []
                current_size = 0.0

            current_bucket.append(param)
            current_size += param_bytes

        # Last bucket
        if current_bucket:
            bucket_idx = len(self._buckets)
            self._buckets.append(current_bucket)
            self._bucket_pending.append(len(current_bucket))
            for p in current_bucket:
                self._param_to_bucket[id(p)] = bucket_idx

        # ── Register hooks ────────────────────────────────────────────────────
        for param in reversed_params:
            param.register_post_accumulate_grad_hook(self._make_hook(param))

    def _make_hook(self, target_param: nn.Parameter):
        """Create a hook that decrements bucket pending count and fires all-reduce when ready."""
        bucket_idx = self._param_to_bucket[id(target_param)]

        def hook(param: torch.Tensor):
            self._bucket_pending[bucket_idx] -= 1
            if self._bucket_pending[bucket_idx] == 0:
                self._allreduce_bucket(bucket_idx)
        return hook

    def _allreduce_bucket(self, bucket_idx: int):
        """Flatten bucket grads, launch async all-reduce."""
        from torch._utils import _flatten_dense_tensors

        bucket_params = self._buckets[bucket_idx]
        grads = [p.grad for p in bucket_params]
        flat = _flatten_dense_tensors(grads)
        handle = dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=True)
        self._handles.append((handle, bucket_idx, flat))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        """Wait on all bucket all-reduces, average, and scatter back to param.grad."""
        from torch._utils import _unflatten_dense_tensors

        world_size = dist.get_world_size()
        for handle, bucket_idx, flat in self._handles:
            handle.wait()
            flat /= world_size
            bucket_params = self._buckets[bucket_idx]
            grads = [p.grad for p in bucket_params]
            for p, unflat in zip(bucket_params, _unflatten_dense_tensors(flat, grads)):
                p.grad.copy_(unflat)
        self._handles.clear()

        # Reset pending counts for next iteration
        for i in range(len(self._buckets)):
            self._bucket_pending[i] = len(self._buckets[i])

"""
Limitations of theis version of DDP:
1.It conducts a separate all-reduce operation for every parameter tensor. Each communication call incurs
overhead, so it may be advantageous to batch communication calls to minimize this overhead.

2. It waits for the backward pass to finish before communicating gradients. However, the backward pass
is incrementally computed. Thus, when a parameter gradient is ready, it can immediately be commu-
nicated without waiting for the gradients of the other parameters. This allows us to overlap communi-
cation of gradients with computation of the backward pass, reducing the overhead of distributed data
parallel training
"""

"""
for limitation 2, we apply the pipeline as below:
Backward:  [Layer N 梯度] [Layer N-1 梯度] [Layer N-2 梯度] ... [Layer 1 梯度]
                ↓               ↓                ↓                  ↓
通信:      [async AR N]   [async AR N-1]   [async AR N-2]    [async AR 1]
                ←──── 与 backward 并行执行 ────→
finish_gradient_synchronization():  等所有 handle 完成
Optimizer step

=================================================================
  DDP Overlap Benchmark | model=small | world_size=2
  bs/rank=4 | seq=512 | mp=False
  device=cuda:0 | backend=gloo
=================================================================
  params=128,625,408 (111 tensors)

── naive: sync per-parameter all-reduce (111 calls) ──
  [      naive] iter  1: total=1734.5ms  comm=1001.2ms  comm%=57.7%
  [      naive] iter  2: total=1789.7ms  comm=1025.9ms  comm%=57.3%
  [      naive] iter  3: total=1902.3ms  comm=1037.4ms  comm%=54.5%
  [      naive] iter  4: total=1578.4ms  comm=735.2ms  comm%=46.6%
  [      naive] iter  5: total=1534.0ms  comm=913.5ms  comm%=59.5%
  [      naive] iter  6: total=1583.6ms  comm=856.3ms  comm%=54.1%
  [      naive] iter  7: total=1634.3ms  comm=990.3ms  comm%=60.6%
  [      naive] iter  8: total=2011.2ms  comm=1311.7ms  comm%=65.2%
  [      naive] iter  9: total=1962.7ms  comm=1345.7ms  comm%=68.6%
  [      naive] iter 10: total=1852.9ms  comm=1290.9ms  comm%=69.7%

── flat: single all-reduce (1 call) ──
  [       flat] iter  1: total=1973.1ms  comm=1266.0ms  comm%=64.2%
  [       flat] iter  2: total=1692.4ms  comm=1013.8ms  comm%=59.9%
  [       flat] iter  3: total=1542.4ms  comm=820.4ms  comm%=53.2%
  [       flat] iter  4: total=1998.2ms  comm=1245.9ms  comm%=62.4%
  [       flat] iter  5: total=1588.1ms  comm=853.5ms  comm%=53.7%
  [       flat] iter  6: total=2058.8ms  comm=1318.9ms  comm%=64.1%
  [       flat] iter  7: total=1563.5ms  comm=900.3ms  comm%=57.6%
  [       flat] iter  8: total=1473.3ms  comm=729.4ms  comm%=49.5%
  [       flat] iter  9: total=2076.3ms  comm=1318.6ms  comm%=63.5%
  [       flat] iter 10: total=1772.5ms  comm=1077.5ms  comm%=60.8%

── overlapping: async per-parameter via hooks ──
  [overlapping] iter  1: total=847.3ms  wait=751.6ms  wait%=88.7%
  [overlapping] iter  2: total=995.0ms  wait=897.5ms  wait%=90.2%
  [overlapping] iter  3: total=988.8ms  wait=890.6ms  wait%=90.1%
  [overlapping] iter  4: total=789.4ms  wait=689.1ms  wait%=87.3%
  [overlapping] iter  5: total=661.6ms  wait=565.0ms  wait%=85.4%
  [overlapping] iter  6: total=597.7ms  wait=499.5ms  wait%=83.6%
  [overlapping] iter  7: total=586.3ms  wait=489.2ms  wait%=83.4%
  [overlapping] iter  8: total=581.6ms  wait=484.7ms  wait%=83.3%
  [overlapping] iter  9: total=655.8ms  wait=557.4ms  wait%=85.0%
  [overlapping] iter 10: total=580.9ms  wait=484.6ms  wait%=83.4%

=================================================================
  RESULTS (small, world_size=2)
=================================================================
  Method           Total ms   Comm/Wait ms  Overhead%
  ────────────── ────────── ────────────── ──────────
  naive              1758.4         1050.8      59.8%
  flat               1773.9         1054.4      59.4%
  overlapping         728.4          630.9      86.6%
  ────────────── ────────── ────────────── ──────────
  Speedup vs naive:  flat=0.99x  overlapping=2.41x
  Comm reduction:    flat=1.00x  overlapping=1.67x
=================================================================
(base_copy) yuyang@gpu-server:~/Experiment/experiment/assignment2-systems$ CUDA_VISIBLE_DEVICES=3,4 uv run python Experiment/ddp_overlap_benchmarking.py

=================================================================
  DDP Overlap Benchmark | model=small | world_size=2
  bs/rank=4 | seq=512 | mp=False
  device=cuda:0 | backend=gloo
=================================================================
  params=128,625,408 (111 tensors)

── naive: sync per-parameter all-reduce (111 calls) ──
  [      naive] iter  1: total=789.7ms  comm=695.6ms  comm%=88.1%
  [      naive] iter  2: total=816.1ms  comm=721.6ms  comm%=88.4%
  [      naive] iter  3: total=953.4ms  comm=857.6ms  comm%=90.0%
  [      naive] iter  4: total=920.2ms  comm=824.5ms  comm%=89.6%
  [      naive] iter  5: total=901.8ms  comm=807.2ms  comm%=89.5%
  [      naive] iter  6: total=943.0ms  comm=848.1ms  comm%=89.9%
  [      naive] iter  7: total=922.6ms  comm=827.8ms  comm%=89.7%
  [      naive] iter  8: total=905.2ms  comm=810.8ms  comm%=89.6%
  [      naive] iter  9: total=985.4ms  comm=891.4ms  comm%=90.5%
  [      naive] iter 10: total=998.0ms  comm=903.9ms  comm%=90.6%

── flat: single all-reduce (1 call) ──
  [       flat] iter  1: total=710.4ms  comm=616.1ms  comm%=86.7%
  [       flat] iter  2: total=800.8ms  comm=706.7ms  comm%=88.2%
  [       flat] iter  3: total=702.3ms  comm=608.5ms  comm%=86.6%
  [       flat] iter  4: total=789.8ms  comm=695.7ms  comm%=88.1%
  [       flat] iter  5: total=721.9ms  comm=627.6ms  comm%=86.9%
  [       flat] iter  6: total=792.0ms  comm=698.1ms  comm%=88.1%
  [       flat] iter  7: total=708.1ms  comm=614.0ms  comm%=86.7%
  [       flat] iter  8: total=747.1ms  comm=653.2ms  comm%=87.4%
  [       flat] iter  9: total=687.4ms  comm=593.3ms  comm%=86.3%
  [       flat] iter 10: total=726.6ms  comm=632.5ms  comm%=87.1%

── overlapping: async per-parameter via hooks ──
  [overlapping] iter  1: total=810.4ms  wait=714.7ms  wait%=88.2%
  [overlapping] iter  2: total=818.8ms  wait=723.3ms  wait%=88.3%
  [overlapping] iter  3: total=840.9ms  wait=745.0ms  wait%=88.6%
  [overlapping] iter  4: total=777.9ms  wait=681.8ms  wait%=87.6%
  [overlapping] iter  5: total=806.4ms  wait=710.7ms  wait%=88.1%
  [overlapping] iter  6: total=711.9ms  wait=616.4ms  wait%=86.6%
  [overlapping] iter  7: total=788.6ms  wait=693.0ms  wait%=87.9%
  [overlapping] iter  8: total=787.9ms  wait=691.7ms  wait%=87.8%
  [overlapping] iter  9: total=831.2ms  wait=734.3ms  wait%=88.3%
  [overlapping] iter 10: total=860.8ms  wait=763.8ms  wait%=88.7%

=================================================================
  RESULTS (small, world_size=2)
=================================================================
  Method           Total ms   Comm/Wait ms  Overhead%
  ────────────── ────────── ────────────── ──────────
  naive               913.5          818.9      89.6%
  flat                738.6          644.6      87.3%
  overlapping         803.5          707.5      88.0%
  ────────────── ────────── ────────────── ──────────
  Speedup vs naive:  flat=1.24x  overlapping=1.14x
  Comm reduction:    flat=1.27x  overlapping=1.16x
=================================================================


bucketed DDP results:
======================================================================
  DDP Bucketed Benchmark | model=small | world_size=2
  bs/rank=4 | seq=512 | mp=False
  device=cuda:0 | backend=gloo
======================================================================
  params=128,625,408 (111 tensors, ~490.7 MB)

── naive: sync per-parameter ──
  [           naive] iter  1: total=932.1ms  comm=837.2ms  comm%=89.8%
  [           naive] iter  2: total=924.0ms  comm=829.4ms  comm%=89.8%
  [           naive] iter  3: total=991.6ms  comm=894.3ms  comm%=90.2%
  [           naive] iter  4: total=941.5ms  comm=845.0ms  comm%=89.8%
  [           naive] iter  5: total=919.1ms  comm=823.4ms  comm%=89.6%
  [           naive] iter  6: total=892.8ms  comm=798.4ms  comm%=89.4%
  [           naive] iter  7: total=962.0ms  comm=866.8ms  comm%=90.1%
  [           naive] iter  8: total=855.7ms  comm=760.9ms  comm%=88.9%
  [           naive] iter  9: total=942.1ms  comm=847.5ms  comm%=90.0%
  [           naive] iter 10: total=941.8ms  comm=847.3ms  comm%=90.0%

── flat: single all-reduce ──
  [            flat] iter  1: total=726.7ms  comm=632.6ms  comm%=87.0%
  [            flat] iter  2: total=620.6ms  comm=526.3ms  comm%=84.8%
  [            flat] iter  3: total=758.1ms  comm=663.4ms  comm%=87.5%
  [            flat] iter  4: total=634.9ms  comm=540.5ms  comm%=85.1%
  [            flat] iter  5: total=821.2ms  comm=726.6ms  comm%=88.5%
  [            flat] iter  6: total=638.6ms  comm=544.0ms  comm%=85.2%
  [            flat] iter  7: total=803.4ms  comm=709.2ms  comm%=88.3%
  [            flat] iter  8: total=656.4ms  comm=562.2ms  comm%=85.7%
  [            flat] iter  9: total=776.1ms  comm=682.1ms  comm%=87.9%
  [            flat] iter 10: total=660.2ms  comm=565.4ms  comm%=85.6%

── overlap-individual: async per-parameter ──
  [   overlap-indiv] iter  1: total=713.4ms  wait=617.1ms  wait%=86.5%
  [   overlap-indiv] iter  2: total=788.3ms  wait=692.2ms  wait%=87.8%
  [   overlap-indiv] iter  3: total=737.6ms  wait=640.8ms  wait%=86.9%
  [   overlap-indiv] iter  4: total=693.1ms  wait=596.4ms  wait%=86.1%
  [   overlap-indiv] iter  5: total=759.7ms  wait=663.2ms  wait%=87.3%
  [   overlap-indiv] iter  6: total=695.7ms  wait=599.5ms  wait%=86.2%
  [   overlap-indiv] iter  7: total=774.3ms  wait=678.0ms  wait%=87.6%
  [   overlap-indiv] iter  8: total=750.4ms  wait=653.3ms  wait%=87.1%
  [   overlap-indiv] iter  9: total=751.0ms  wait=653.7ms  wait%=87.0%
  [   overlap-indiv] iter 10: total=691.1ms  wait=594.5ms  wait%=86.0%

── bucket-1MB ──
  (111 buckets)
  [      bucket-1MB] iter  1: total=789.5ms  wait=692.8ms  wait%=87.7%
  [      bucket-1MB] iter  2: total=940.2ms  wait=844.1ms  wait%=89.8%
  [      bucket-1MB] iter  3: total=687.5ms  wait=590.3ms  wait%=85.9%
  [      bucket-1MB] iter  4: total=818.6ms  wait=721.9ms  wait%=88.2%
  [      bucket-1MB] iter  5: total=697.6ms  wait=601.8ms  wait%=86.3%
  [      bucket-1MB] iter  6: total=849.1ms  wait=752.3ms  wait%=88.6%
  [      bucket-1MB] iter  7: total=812.7ms  wait=715.4ms  wait%=88.0%
  [      bucket-1MB] iter  8: total=883.3ms  wait=787.4ms  wait%=89.1%
  [      bucket-1MB] iter  9: total=847.8ms  wait=750.9ms  wait%=88.6%
  [      bucket-1MB] iter 10: total=761.2ms  wait=663.9ms  wait%=87.2%

── bucket-10MB ──
  (50 buckets)
  [     bucket-10MB] iter  1: total=695.5ms  wait=598.8ms  wait%=86.1%
  [     bucket-10MB] iter  2: total=716.6ms  wait=619.6ms  wait%=86.5%
  [     bucket-10MB] iter  3: total=704.9ms  wait=608.1ms  wait%=86.3%
  [     bucket-10MB] iter  4: total=752.0ms  wait=654.9ms  wait%=87.1%
  [     bucket-10MB] iter  5: total=708.5ms  wait=611.3ms  wait%=86.3%
  [     bucket-10MB] iter  6: total=687.2ms  wait=590.5ms  wait%=85.9%
  [     bucket-10MB] iter  7: total=715.2ms  wait=618.2ms  wait%=86.4%
  [     bucket-10MB] iter  8: total=620.0ms  wait=522.8ms  wait%=84.3%
  [     bucket-10MB] iter  9: total=697.6ms  wait=600.1ms  wait%=86.0%
  [     bucket-10MB] iter 10: total=705.1ms  wait=608.1ms  wait%=86.2%

── bucket-100MB ──
  (6 buckets)
  [    bucket-100MB] iter  1: total=656.0ms  wait=555.5ms  wait%=84.7%
  [    bucket-100MB] iter  2: total=656.8ms  wait=555.8ms  wait%=84.6%
  [    bucket-100MB] iter  3: total=677.6ms  wait=576.5ms  wait%=85.1%
  [    bucket-100MB] iter  4: total=692.4ms  wait=592.0ms  wait%=85.5%
  [    bucket-100MB] iter  5: total=690.5ms  wait=590.4ms  wait%=85.5%
  [    bucket-100MB] iter  6: total=708.7ms  wait=608.7ms  wait%=85.9%
  [    bucket-100MB] iter  7: total=719.7ms  wait=619.7ms  wait%=86.1%
  [    bucket-100MB] iter  8: total=635.5ms  wait=535.7ms  wait%=84.3%
  [    bucket-100MB] iter  9: total=654.6ms  wait=554.7ms  wait%=84.7%
  [    bucket-100MB] iter 10: total=644.2ms  wait=544.5ms  wait%=84.5%

── bucket-1000MB ──
  (1 buckets)
  [   bucket-1000MB] iter  1: total=723.2ms  wait=607.6ms  wait%=84.0%
  [   bucket-1000MB] iter  2: total=738.6ms  wait=622.5ms  wait%=84.3%
  [   bucket-1000MB] iter  3: total=751.7ms  wait=636.3ms  wait%=84.6%
  [   bucket-1000MB] iter  4: total=618.7ms  wait=503.0ms  wait%=81.3%
  [   bucket-1000MB] iter  5: total=714.5ms  wait=598.8ms  wait%=83.8%
  [   bucket-1000MB] iter  6: total=666.1ms  wait=550.5ms  wait%=82.6%
  [   bucket-1000MB] iter  7: total=755.7ms  wait=640.2ms  wait%=84.7%
  [   bucket-1000MB] iter  8: total=674.8ms  wait=559.3ms  wait%=82.9%
  [   bucket-1000MB] iter  9: total=718.0ms  wait=602.4ms  wait%=83.9%
  [   bucket-1000MB] iter 10: total=655.3ms  wait=539.9ms  wait%=82.4%

======================================================================
  RESULTS (small, world_size=2)
======================================================================
  Method               Total ms   Comm/Wait ms  Overhead%   vs naive
  ────────────────── ────────── ────────────── ────────── ──────────
  naive                   930.3          835.0      89.8%      1.00x
  flat                    709.6          615.2      86.7%      1.31x
  overlap-indiv           735.5          638.9      86.9%      1.26x
  bucket-1MB              808.7          712.1      88.0%      1.15x
  bucket-10MB             700.3          603.2      86.1%      1.33x
  bucket-100MB            673.6          573.3      85.1%      1.38x
  bucket-1000MB           701.7          586.0      83.5%      1.33x
======================================================================
"""