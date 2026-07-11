from __future__ import annotations

import asyncio
import json

from typing import Any

import wandb
from datasets import load_dataset

from scale_rl.trainer.setup_utils import print0
from scale_rl.envs.dapo_env import check_answer, extract_last_boxed
from scale_rl.inference.rollout_worker import vLLMRollout
from scale_rl.eval.utils import pass_at_k

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful competition math solver. "
    "Think step by step before answering. "
    "Then provide the final answer inside \\boxed{...}."
)

# DAPO-style math benchmarks this evaluator supports. Not valid for
# livecodebench, whose scoring requires running generated code against
# test cases rather than comparing a single boxed answer.
DATASET_CONFIGS = {
    "aime_2024": {
        "hf_path": "HuggingFaceH4/aime_2024",
        "split": "train",
        "problem_col": "problem",
        "answer_col": "answer",
        "id_col": "id",
    },
    "aime_2025": {
        "hf_path": "MathArena/aime_2025",
        "split": "train",
        "problem_col": "problem",
        "answer_col": "answer",
        "id_col": "problem_idx",
    },
    "hmmt_2025": {
        "hf_path": "MathArena/hmmt_feb_2025",
        "split": "train",
        "problem_col": "problem",
        "answer_col": "answer",
        "id_col": "problem_idx",
    },
}


async def _run_eval_one_dataset(
    dataset_name: str,
    rollout_worker: vLLMRollout,
    eval_k: int,
    eval_max_tokens: int,
    step: int,
    temperature: float,
    top_k: int,
) -> dict[str, float]:
    """Evaluates the already weight-synced rollout worker on one benchmark.

    Args:
        dataset_name: Key into `DATASET_CONFIGS`.
        rollout_worker: Weight-synced vLLM rollout worker to generate from.
        eval_k: Samples per problem, for pass@k.
        eval_max_tokens: Max generation length.
        step: Trainer step, used for logging and the W&B x-axis.
        temperature: Sampling temperature.
        top_k: Sampling top-k (-1 disables it).

    Returns:
        A dict with a single `eval/{dataset_name}/pass@{eval_k}` entry.
    """
    cfg = DATASET_CONFIGS[dataset_name]
    problems: list[dict] = list(
        await asyncio.to_thread(load_dataset, cfg["hf_path"], split=cfg["split"])
    )
    prompts = [p[cfg["problem_col"]] for p in problems]

    sampling_params = {"max_tokens": eval_max_tokens, "temperature": temperature, "top_k": top_k, "logprobs": 1}
    rollouts = await rollout_worker.generate_batch(
        prompts, eval_k, sampling_params, system_prompt=DEFAULT_SYSTEM_PROMPT
    )

    # rollouts is flat: eval_k responses per problem in order
    per_problem = []
    for i, prob in enumerate(problems):
        batch = rollouts[i * eval_k : (i + 1) * eval_k]
        preds = [extract_last_boxed(r.response) for r in batch]
        n_correct = sum(check_answer(p, prob[cfg["answer_col"]]) for p in preds)
        per_problem.append(
            {
                "problem_idx": prob[cfg["id_col"]],
                "n_correct": n_correct,
                "pass_at_k": pass_at_k(eval_k, n_correct, eval_k),
            }
        )

    overall = sum(r["pass_at_k"] for r in per_problem) / len(per_problem)
    metrics = {f"eval/{dataset_name}/pass@{eval_k}": overall}

    print0(f"[eval step={step}] {dataset_name}: {json.dumps(metrics)}")
    for i, (r, prob) in enumerate(zip(per_problem, problems)):
        batch = rollouts[i * eval_k : (i + 1) * eval_k]
        print0(f"  problem {r['problem_idx']}: {r['n_correct']}/{eval_k}  pass@{eval_k}={r['pass_at_k']:.3f}")
        print0(f"    question: {prob[cfg['problem_col']]}")
        for j, ro in enumerate(batch):
            print0(f"    sample {j}: {ro.response}")

    if wandb.run is not None:
        wandb.log(metrics, step=step)

    return metrics


def run_eval(
    rollout_worker: vLLMRollout,
    eval_k: int,
    eval_max_tokens: int,
    step: int,
    eval_datasets: str = "aime_2025",
    temperature: float = 0.6,
    top_k: int = -1,
) -> dict[str, Any]:
    """Evaluates on one or more DAPO-style math benchmarks (AIME/HMMT).

    Only valid for the DAPO dataset's evaluation slot, not livecodebench.

    Args:
        rollout_worker: Weight-synced vLLM rollout worker to generate from.
        eval_k: Samples per problem, for pass@k.
        eval_max_tokens: Max generation length.
        step: Trainer step, used for logging and the W&B x-axis.
        eval_datasets: Comma-separated `DATASET_CONFIGS` keys, e.g.
          "aime_2025,aime_2024,hmmt_2025".
        temperature: Sampling temperature.
        top_k: Sampling top-k (-1 disables it).

    Returns:
        Combined `eval/{dataset_name}/pass@{eval_k}` metrics from every
        dataset in `eval_datasets`.
    """

    async def _run_all() -> list[dict[str, float]]:
        return await asyncio.gather(*[
            _run_eval_one_dataset(
                dataset_name=dataset_name.strip(),
                rollout_worker=rollout_worker,
                eval_k=eval_k,
                eval_max_tokens=eval_max_tokens,
                step=step,
                temperature=temperature,
                top_k=top_k,
            )
            for dataset_name in eval_datasets.split(",")
        ])

    metrics: dict[str, Any] = {}
    for dataset_metrics in asyncio.run(_run_all()):
        metrics.update(dataset_metrics)
    return metrics
