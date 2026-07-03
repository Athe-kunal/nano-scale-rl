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
    gradient_checkpointing: bool = True
    gradient_accumulation_steps: int = 1

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
    # Sharding strategy: no_shard | full_shard | shard_grad_op | hybrid_shard | hybrid_shard_zero2
    #   no_shard          – no sharding (single GPU or DDP-equivalent); parameters keep original names,
    #                       compatible with vLLM weight transfer
    #   full_shard        – shard params + grads + optimizer states across all ranks (ZeRO-3)
    #   shard_grad_op     – shard grads + optimizer states only, replicate params (ZeRO-2)
    #   hybrid_shard      – full_shard within node, replicate across nodes; requires 2D device mesh
    #   hybrid_shard_zero2– shard_grad_op within node, replicate across nodes; requires 2D device mesh
    fsdp_sharding_strategy: str = "no_shard"
    sync_module_states: bool = True

    # NCCL weight transfer
    master_address: str = "127.0.0.1"
    master_port: int = 29600          # torchrun rendezvous port
    weight_transfer_port: int = 29601 # separate port for vLLM StatelessProcessGroup rendezvous
    weight_transfer_packed: bool = False  # packed=True incompatible with torch 2.10 ProcessGroup.broadcast
    vllm_clear_kv_cache: bool = False     # True: abort in-flight rollouts + clear KV cache on sync (correct, wasteful)

    # Training loop
    total_steps: int = 1000
    log_every: int = 10
    high_pass_rate_threshold: float = 0.9

    # Evaluation (AIME 2025, via scale_rl.eval.eval_aime_2025.run_eval)
    eval_every: int = 0        # evaluate every N trainer steps (0 = disabled)
    eval_k: int = 4            # samples per problem, for pass@k
    eval_max_tokens: int = 2048
    eval_temperature: float = 0.6
    eval_top_k: int = -1

    # Wandb
    wandb_project: str = "ScaleRL"
    wandb_run_name: str = ""   # empty = wandb auto-names the run
    wandb_enabled: bool = True