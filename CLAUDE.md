# nano-scale-rl

## What this is

This is a **pedagogical codebase**, not a production library. Its purpose is to
teach two things at once, using RLHF/RLVR-style post-training of language
models as the vehicle:

1. **Distributed computing**: FSDP training, NCCL weight transfer between a
   trainer process and a vLLM rollout worker, multi-GPU orchestration.
2. **Reinforcement learning for language models**: rollout generation, reward
   computation, replay buffers, policy-gradient losses.

The user is here to *learn*, not just to ship a working feature. Optimize for
understanding over speed of delivery.

## How to collaborate on this codebase

- **Reserve probing questions for distributed-training and RLVR/RLHF
  concepts** — e.g. why weights are broadcast via NCCL process groups rather
  than saved/loaded, why KV cache needs to be paused/cleared during a weight
  update, why an HTTP call and a collective op need to run concurrently,
  why log probs are chunked, why rank offsets matter, why on-policy vs
  off-policy staleness matters for the RL objective. Ask before explaining;
  don't assume familiarity.
- **Do NOT ask the user to make mechanical/wiring decisions** (which file to
  edit next, which endpoint name is correct, how to thread a parameter
  through a call chain, whether to fix a bug now or later). Investigate,
  decide, and implement those directly — that's plumbing, not the lesson.
- **Small, incremental code changes.** Don't deliver large multi-file
  refactors in one shot. Change one function or one small piece at a time,
  and explain *why* the change works the way it does — but don't stop to ask
  permission at each mechanical step once the direction is set.
- Surface subtle bugs or stale code (like unused config flags, mismatched
  endpoint names, or a launcher invoking a module that no longer has an
  entrypoint) as a discussion point when they touch a distributed/RL
  concept worth understanding — but don't block small fixes on approval.

## Code style

- Follow **Google Python Style Guide** conventions: docstrings in Google
  format (`Args:`, `Returns:`, `Raises:`), 4-space indentation, `snake_case`
  for functions/variables, `CapWords` for classes, module-level constants in
  `ALL_CAPS`. Avoid overly clever one-liners; prefer readable, explicit code
  — this is a teaching codebase, clarity beats brevity.
- Type hints on public function signatures.
- No unnecessary abstractions — this codebase favors directly readable code
  over generic frameworks, since the point is for the user to be able to
  trace execution end to end.

## Repo layout

- `scale_rl/trainer/` — FSDP training loop, distributed setup utilities.
- `scale_rl/inference/rollout_worker.py` — async HTTP client used by the
  trainer to talk to the vLLM rollout server (generation + NCCL weight
  transfer control endpoints).
- `scale_rl/algos/` — RL losses (policy gradient / PPO-style) and helpers.
- `scale_rl/envs/` — task environments (e.g. LiveCodeBench, DAPO) that
  produce prompts and score rollouts.
- `scale_rl/replay_buffer/` — buffer for storing/sampling rollouts.
- `scale_rl/eval/` — evaluation harness (e.g. AIME 2025).
- `scale_rl/scripts/train.sh` / `train.yaml` / `train_entry.py` — launches
  the vLLM rollout worker and the FSDP trainer as separate processes.

Note: this repo has moved fast and has some drift between `train.sh` /
`train.yaml` and the current `rollout_worker.py` implementation (e.g. CLI
flags that no longer correspond to an actual argparse entrypoint). Treat
config/script mismatches as things worth flagging to the user, since
understanding *why* they're stale is itself a useful lesson in reading a
codebase critically.
