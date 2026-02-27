"""Optimizer State Sharding.

Each rank's optimizer only maintains state for ~1/world_size of the parameters.
After each optimizer step, updated parameters are broadcast to all ranks.
"""

from typing import Any, Type

import torch
import torch.distributed as dist
import torch.optim as optim


class ShardedOptimizer(optim.Optimizer):
    """Optimizer wrapper that shards optimizer state across DDP ranks.

    Each rank only holds optimizer state (e.g., Adam's m and v buffers) for
    its assigned subset of parameters. After step(), updated parameters are
    broadcast from the owning rank to all other ranks.
    """

    def __init__(self, params, optimizer_cls: Type[optim.Optimizer], **kwargs: Any):
        self._all_params: list[torch.nn.Parameter] = []
        self._param_to_rank: dict[int, int] = {}
        self._optimizer_cls = optimizer_cls
        self._optimizer_kwargs = kwargs
        self._inner_optimizer: optim.Optimizer | None = None
        self._world_size = dist.get_world_size()
        self._rank = dist.get_rank()

        # super().__init__ calls self.add_param_group for each group
        super().__init__(params, defaults=kwargs)

        # Build inner optimizer with only this rank's parameters
        my_params = [p for p in self._all_params if self._param_to_rank[id(p)] == self._rank]
        if my_params:
            self._inner_optimizer = optimizer_cls(my_params, **kwargs)
        else:
            self._inner_optimizer = None

    def add_param_group(self, param_group: dict[str, Any]):
        """Assign parameters in the group to ranks in round-robin order."""
        # Let super handle standard bookkeeping
        super().add_param_group(param_group)

        # Assign each new unique parameter to a rank
        for param in param_group["params"]:
            if id(param) in self._param_to_rank:
                continue
            idx = len(self._all_params)
            self._param_to_rank[id(param)] = idx % self._world_size
            self._all_params.append(param)

        # If inner optimizer already exists (post-init call), update it
        if self._inner_optimizer is not None:
            my_new = [p for p in param_group["params"]
                      if self._param_to_rank[id(p)] == self._rank]
            if my_new:
                self._inner_optimizer.add_param_group(
                    {k: v for k, v in param_group.items() if k != "params"} | {"params": my_new}
                )

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        """Step the inner optimizer, then broadcast updated params."""
        if self._inner_optimizer is not None:
            self._inner_optimizer.step(closure=closure, **kwargs)

        # Synchronize: each param is broadcast from its owning rank
        for param in self._all_params:
            dist.broadcast(param.data, src=self._param_to_rank[id(param)])

    def zero_grad(self, set_to_none: bool = True):
        """Zero all gradients (not just this rank's shard)."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()

    @property
    def state(self):
        if self._inner_optimizer is not None:
            return self._inner_optimizer.state
        return {}

    @state.setter
    def state(self, value):
        if self._inner_optimizer is not None:
            self._inner_optimizer.state = value

"""
Model: medium, params=423,183,360 (~1614.3 MB fp32)
AdamW m+v: non-sharded=3228.6 MB, sharded(1/4)=807.2 MB
Expected savings: ~2421.4 MB
Using 4 GPUs

======= Running NON-SHARDED ======

============================================================
  NON-SHARDED | model=medium | ws=4
  params=423,183,360 (~1614.3 MB in fp32)
============================================================

────────────────────────────────────────────────────────────
  NON-SHARDED optimizer
────────────────────────────────────────────────────────────
  After model init:       alloc= 1619.2 MB  peak= 1619.2 MB
  After optimizer init:   alloc= 1619.2 MB  peak= 1619.2 MB
  Before optimizer step:  alloc= 6559.4 MB  peak=14057.7 MB
  After optimizer step:   alloc= 6559.4 MB  peak=14057.7 MB
    iter 1: 2321.4 ms
    iter 2: 2721.1 ms
    iter 3: 2723.5 ms
    iter 4: 2633.4 ms
    iter 5: 2501.5 ms

  Avg step time:  2580.2 ms
  Overall peak allocated: 14057.7 MB

====== Running SHARDED ======

============================================================
  SHARDED | model=medium | ws=4
  params=423,183,360 (~1614.3 MB in fp32)
============================================================

────────────────────────────────────────────────────────────
  SHARDED optimizer
────────────────────────────────────────────────────────────
  After model init:       alloc= 1619.2 MB  peak= 1619.2 MB
  After optimizer init:   alloc= 1619.2 MB  peak= 1619.2 MB
  Before optimizer step:  alloc= 4175.1 MB  peak=11673.4 MB
  After optimizer step:   alloc= 4175.1 MB  peak=11673.4 MB
    iter 1: 5188.5 ms
    iter 2: 5292.2 ms
    iter 3: 5127.7 ms
    iter 4: 4886.7 ms
    iter 5: 4940.2 ms

  Avg step time:  5087.1 ms
  Overall peak allocated: 11673.4 MB
"""