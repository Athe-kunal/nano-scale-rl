from typing import Type
import functools
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

_STRATEGY_MAP: dict[str, ShardingStrategy] = {
    "no_shard":           ShardingStrategy.NO_SHARD,
    "full_shard":         ShardingStrategy.FULL_SHARD,
    "shard_grad_op":      ShardingStrategy.SHARD_GRAD_OP,
    "hybrid_shard":       ShardingStrategy.HYBRID_SHARD,
    "hybrid_shard_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
}

def _param_init_fn(module: nn.Module):
    module.to_empty(device=torch.cuda.current_device(), recurse=False)

def prepare_dp_model(
    model: nn.Module,
    dtype: str,
    sync_module_states: bool,
    device_mesh: dist.DeviceMesh,
    sharding_strategy: str = "no_shard",
) -> nn.Module:
    if sharding_strategy not in _STRATEGY_MAP:
        raise ValueError(
            f"Unknown fsdp_sharding_strategy {sharding_strategy!r}. "
            f"Choose from: {list(_STRATEGY_MAP)}"
        )

    # NO_SHARD: skip FSDP entirely.  FSDP internally renames all parameters to
    # _flat_param even with NO_SHARD, which breaks vLLM's weight-name lookup.
    # A plain model on the correct device is equivalent for single-GPU training.
    if sharding_strategy == "no_shard":
        device = torch.device("cuda", torch.cuda.current_device())
        return model.to(device=device, dtype=getattr(torch, dtype))

    strategy = _STRATEGY_MAP[sharding_strategy]

    def get_module_cls_from_name(name: str) -> Type[nn.Module]:
        for module in model.modules():
            if module.__class__.__name__ == name:
                return module.__class__

    transformer_layer_cls = {
        get_module_cls_from_name(name)
        for name in model._no_split_modules
    }
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_layer_cls
    )

    dtype_torch: torch.dtype = getattr(torch, dtype)
    mixed_precision = MixedPrecision(
        param_dtype=dtype_torch,
        reduce_dtype=dtype_torch,
        buffer_dtype=dtype_torch,
    )

    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=strategy,
        mixed_precision=mixed_precision,
        param_init_fn=_param_init_fn,
        sync_module_states=sync_module_states,
        device_mesh=device_mesh,
        device_id=torch.cuda.current_device()
    )