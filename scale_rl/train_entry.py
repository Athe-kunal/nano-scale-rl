"""
torchrun entrypoint for pipeline RL training.

Reads the OmegaConf yaml, initialises distributed state, builds the FSDP
Trainer, and calls trainer.train().

Launch via train.sh (which also starts the vLLM rollout worker).
"""

from __future__ import annotations

import argparse
import torch.distributed as dist

from omegaconf import OmegaConf
from loguru import logger

from scale_rl.trainer.setup_utils import compute_init, compute_cleanup
from scale_rl.trainer.config import TrainerConfig
from scale_rl.trainer.train import Trainer
from scale_rl.envs import DATASET_ENV_CLS


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
    env_cls = DATASET_ENV_CLS[cfg.dataset]

    # ---- device mesh for FSDP ----
    # Single-node: 1D mesh over all trainer GPUs → FULL_SHARD.
    # Multi-node:  2D mesh (num_nodes, gpus_per_node) → HYBRID_SHARD.
    # dp_utils.py picks the right ShardingStrategy based on mesh.ndim.
    device_mesh = dist.device_mesh.init_device_mesh("cuda", (world_size,))

    # ---- dataset ----
    if cfg.dataset == "dapo":
        dataset_id = raw_yaml.get("dataset_id", "open-r1/DAPO-Math-17k-Processed")
        dataset_config = raw_yaml.get("dataset_config", "all")
        dataset_split = raw_yaml.get("dataset_split", "train")
        if rank == 0:
            logger.info(f"Loading dataset {dataset_id}/{dataset_config} (split={dataset_split}) ...")
        records = env_cls.load(dataset_id=dataset_id, config_name=dataset_config, split=dataset_split)
    elif cfg.dataset == "livecodebench":
        dataset_split = raw_yaml.get("dataset_split", "train")
        if rank == 0:
            logger.info(f"Loading livecodebench (split={dataset_split}) ...")
        records = env_cls.load(dataset_split=dataset_split)

    prompts = [env.prompt for env in records]
    envs = records
    if rank == 0:
        logger.info(f"Dataset loaded: {len(records)} examples.")

    # ---- trainer ----
    trainer = Trainer(cfg, device_mesh=device_mesh)

    try:
        trainer.train(envs=envs, prompts=prompts)
    finally:
        compute_cleanup()


if __name__ == "__main__":
    main()
