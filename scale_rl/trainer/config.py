from dataclasses import dataclass
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    # Model
    model_path: str
    dtype: str = "bfloat16"

    # Optimiser
    lr: float = 1e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # RL algorithm + loss kwargs
    algorithm: str = "dapo"       # grpo | dapo | reinforce | gspo | cispo | maxrl
    clip_eps: float = 0.2         # PPO symmetric clip (grpo/gspo)
    clip_low: float = 0.8         # DAPO asymmetric lower bound
    clip_high: float = 1.28       # DAPO asymmetric upper bound
    kl_coef: float = 0.0
    tis_C: float = 0.0            # temporal IS truncation threshold (0 = off)

    # Sequence
    max_seq_len: int = 4096

    # Rollout / buffer
    prompts_per_batch: int = 16
    rollouts_per_step: int = 8    # G in GRPO/DAPO
    stale_steps: int = 1          # sync weights every K trainer updates
    rollout_worker_url: str = "http://127.0.0.1:8047"

    # Generation
    max_new_tokens: int = 1024
    temperature: float = 1.0
    top_k: int = -1

    # FSDP
    sync_module_states: bool = True

    # NCCL weight transfer
    master_address: str = "127.0.0.1"
    master_port: int = 29600
    weight_transfer_packed: bool = True

    # Training loop
    total_steps: int = 1000
    log_every: int = 10
    high_pass_rate_threshold: float = 0.9