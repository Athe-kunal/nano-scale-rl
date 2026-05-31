"""
torchrun entrypoint for pipeline RL training.

Reads the OmegaConf yaml, initialises distributed state, builds the FSDP
Trainer, and calls trainer.train().

Launch via train.sh (which also starts the vLLM rollout worker).
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from loguru import logger

from scale_rl.trainer.common import compute_init, compute_cleanup
from scale_rl.trainer.config import TrainerConfig
from scale_rl.trainer.train import Trainer
from scale_rl.envs.dapo_env import DapoMathEnv


def build_config(yaml_path: str) -> TrainerConfig:
    raw: dict = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True)  # type: ignore[assignment]
    cfg_keys = TrainerConfig.__dataclass_fields__.keys()
    return TrainerConfig(**{k: v for k, v in raw.items() if k in cfg_keys})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to train.yaml")
    args = parser.parse_args()

    # Distributed + device init (sets up NCCL process group).
    is_ddp, rank, local_rank, world_size, device = compute_init(device_type="cuda")

    cfg = build_config(args.config)

    raw_yaml: dict = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)  # type: ignore[assignment]
    dataset_id     = raw_yaml.get("dataset_id",     "open-r1/DAPO-Math-17k-Processed")
    dataset_config = raw_yaml.get("dataset_config", "all")
    dataset_split  = raw_yaml.get("dataset_split",  "train")

    # ---- device mesh for FSDP ----
    device_mesh = dist.device_mesh.init_device_mesh("cuda", (world_size,))

    # ---- NCCL group: trainer ranks + vLLM worker rank ----
    # The vLLM worker registers itself at rank = world_size (rank_offset in
    # remote_vllm_init_weight_transfer).  We include it in the update group.
    all_ranks = list(range(world_size + 1))
    model_update_group = dist.new_group(ranks=all_ranks, backend="nccl")

    # ---- dataset ----
    if rank == 0:
        logger.info(
            "Loading dataset %s/%s (split=%s) ...",
            dataset_id, dataset_config, dataset_split,
        )
    records = DapoMathEnv.load(
        dataset_id=dataset_id,
        config_name=dataset_config,
        split=dataset_split,
    )
    prompts = [env.prompt for env in records]
    envs    = records
    if rank == 0:
        logger.info("Dataset loaded: %d examples.", len(records))

    # ---- trainer ----
    trainer = Trainer(cfg, device_mesh=device_mesh, model_update_group=model_update_group)

    try:
        trainer.train(envs=envs, prompts=prompts)
    finally:
        compute_cleanup()


if __name__ == "__main__":
    main()
