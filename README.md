# nano-scale-rl

A pedagogical codebase for RLHF/RLVR-style post-training of language models,
built to teach two things together:

1. **Distributed computing** — FSDP training, NCCL weight transfer between a
   trainer process and a vLLM rollout worker, multi-GPU orchestration.
2. **RL for language models** — rollout generation, reward computation,
   replay buffers, policy-gradient losses.

It is not a production library; small, readable pieces over generic
frameworks.

## How it works

Training runs as two cooperating processes, started by
`scale_rl/train.sh`:

- **Rollout worker**: a stock `vllm serve <model>` process, started with
  `VLLM_SERVER_DEV_MODE=1` to expose dev-only HTTP endpoints
  (`/pause`, `/resume`, `/update_weights`, `/init_weight_transfer_engine`)
  alongside the normal completions API.
- **FSDP trainer**: launched via `torchrun -m scale_rl.train_entry`.
  Wraps a HF causal LM in FSDP, and each step:
  1. requests rollouts from the vLLM worker over HTTP
     (`scale_rl/inference/rollout_worker.py`, an async HTTP client — not a
     server or vLLM plugin),
  2. scores them with a task environment,
  3. stores `(prompt, rollout, reward)` tuples in a replay buffer
     (`scale_rl/replay_buffer/buffer.py`) until a full batch is ready,
  4. computes a policy-gradient loss and takes a gradient-accumulated step,
  5. every `stale_steps` gradient updates, pushes updated weights to the
     vLLM worker over an NCCL process group (pausing/clearing its KV cache
     around the update).

Both processes share an NCCL rendezvous (`master_port` / `weight_transfer_port`
in `train.yaml`) so weights move trainer → vLLM without checkpointing to disk.

## RL algorithms

`scale_rl/algos/loss.py` implements six policy-gradient variants, selected via
`algorithm` in `train.yaml`: `grpo`, `dapo`, `reinforce`, `gspo`, `cispo`,
`maxrl`. They differ in clipping (symmetric vs. asymmetric) and loss
aggregation (token-mean vs. sequence-mean), and optionally support a KL
penalty (`kl_coef`, grpo/gspo only) and truncated importance sampling
(`tis_C`).

## Tasks and evaluation

- `scale_rl/envs/dapo_env.py` — `DapoMathEnv`, used with
  `open-r1/DAPO-Math-17k-Processed` (the default in `train.yaml`): extracts
  boxed answers and rewards numeric/string matches.
- `scale_rl/envs/livecodebench.py` — a LiveCodeBench-style coding
  environment with subprocess execution and pass@k scoring. Selected via
  `dataset: livecodebench` in `train.yaml` (default is `dapo`).
- `scale_rl/eval/eval_aime_2025.py` — periodic evaluation on AIME 2025
  (`eval_every` in `train.yaml`), computing pass@k via the same vLLM client
  used for training rollouts.

## Configuration

`scale_rl/train.yaml` maps 1:1 onto `TrainerConfig`
(`scale_rl/trainer/config.py`); `train_entry.py` filters the parsed yaml
through `TrainerConfig`'s fields, so unrecognized keys are silently dropped.
`train.sh` re-parses a subset of the same yaml with a small Python snippet to
derive GPU assignments and vLLM launch flags.

## Layout

- `scale_rl/trainer/` — FSDP training loop, distributed setup utilities.
- `scale_rl/inference/rollout_worker.py` — HTTP client the trainer uses to
  talk to the vLLM rollout worker (generation + weight-transfer control).
- `scale_rl/algos/` — policy-gradient losses.
- `scale_rl/envs/` — task environments that produce prompts and score
  rollouts.
- `scale_rl/replay_buffer/` — buffer for storing/sampling rollouts.
- `scale_rl/eval/` — evaluation harness.
- `scale_rl/train.sh` / `train.yaml` / `train_entry.py` — launches
  the vLLM rollout worker and the FSDP trainer as separate processes.
