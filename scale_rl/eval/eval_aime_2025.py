from __future__ import annotations

import asyncio
import json

from typing import Any

import wandb
from datasets import load_dataset

from scale_rl.trainer.setup_utils import print0
from scale_rl.envs.dapo_env import extract_last_boxed
from scale_rl.inference.rollout_worker import vLLMRollout
from scale_rl.eval.utils import pass_at_k

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful competition math solver. "
    "Think step by step before answering. "
    "Then provide the final answer inside \\boxed{...}."
)


def _canon(s: str) -> str:
    return " ".join(str(s).strip().split())


def check_answer(pred: str | None, answer: int | float | str) -> bool:
    if pred is None:
        return False
    pred_s = _canon(pred)
    ans_s = _canon(str(answer))
    if pred_s == ans_s:
        return True
    try:
        return float(pred_s) == float(ans_s)
    except (ValueError, TypeError):
        return False


def run_eval(
    rollout_worker: vLLMRollout,
    eval_k: int,
    eval_max_tokens: int,
    step: int,
    temperature: float = 0.6,
    top_k: int = -1,
) -> dict[str, Any]:
    """Evaluate on AIME 2025 using the already weight-synced rollout worker."""
    problems: list[dict] = list(load_dataset("MathArena/aime_2025", split="train"))
    prompts = [p["problem"] for p in problems]

    sampling_params = {"max_tokens": eval_max_tokens, "temperature": temperature, "top_k": top_k, "logprobs": 1}
    rollouts = asyncio.run(
        rollout_worker.generate_batch(prompts, eval_k, sampling_params, system_prompt=DEFAULT_SYSTEM_PROMPT)
    )

    # rollouts is flat: eval_k responses per problem in order
    per_problem = []
    for i, prob in enumerate(problems):
        batch = rollouts[i * eval_k : (i + 1) * eval_k]
        preds = [extract_last_boxed(r.response) for r in batch]
        n_correct = sum(check_answer(p, prob["answer"]) for p in preds)
        per_problem.append(
            {
                "problem_idx": prob["problem_idx"],
                "n_correct": n_correct,
                "pass_at_k": pass_at_k(eval_k, n_correct, eval_k),
            }
        )

    overall = sum(r["pass_at_k"] for r in per_problem) / len(per_problem)
    metrics = {
        f"eval/pass@{eval_k}": overall,
    }

    print0(f"[eval step={step}] {json.dumps(metrics)}")
    for i, (r, prob) in enumerate(zip(per_problem, problems)):
        batch = rollouts[i * eval_k : (i + 1) * eval_k]
        print0(
            f"  problem {r['problem_idx']:02d}: {r['n_correct']}/{eval_k}  pass@{eval_k}={r['pass_at_k']:.3f}"
        )
        print0(f"    question: {prob['problem']}")
        for j, ro in enumerate(batch):
            print0(f"    sample {j}: {ro.response}")

    if wandb.run is not None:
        wandb.log(metrics, step=step)

    return metrics
