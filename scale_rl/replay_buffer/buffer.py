"""
Replay buffer for pipeline RL (DAPO / GRPO style).

Terminology
-----------
  prompts_per_batch  – number of *distinct* prompts that constitute one
                       trainer update batch.
  rollouts_per_step  – number of sampled completions per prompt (the G
                       factor in GRPO/DAPO).  Total items per batch =
                       prompts_per_batch * rollouts_per_step.
  stale_steps  (K)   – push updated trainer weights into the vLLM worker
                       every K trainer update steps.

Typical call sequence
---------------------
  buf = ReplayBuffer(prompts_per_batch=16, rollouts_per_step=8,
                     stale_steps=4, rollout_worker=worker)
  buf.add(rollouts, rewards)    # called after each vLLM generation round
  while buf.is_ready():
      batch = buf.sample()      # returns one batch and drains it from the buffer
      loss = trainer.step(batch)
      buf.on_trainer_step(step, train_model, model_update_group)
"""

from __future__ import annotations

from collections import deque
from typing import Any

import asyncio

from scale_rl.inference.models import RolloutRecord
from scale_rl.inference.rollout_worker import vLLMRollout

from loguru import logger


class ReplayBuffer:
    """
    FIFO replay buffer that drives the trainer update and vLLM weight-sync
    schedule for pipeline RL.

    Parameters
    ----------
    prompts_per_batch:
        Number of distinct prompts in one trainer batch.
    rollouts_per_step:
        Number of sampled rollouts per prompt (G in GRPO/DAPO).
    stale_steps:
        Sync trainer → vLLM weights every this many trainer update steps.
        Set to 1 for on-policy (sync after every update).
    rollout_worker:
        A ``vLLMRollout`` instance connected to the running vLLM server.
        Required when ``stale_steps > 0``.
    max_size:
        Maximum number of ``RolloutRecord`` objects kept in the buffer.
        Oldest items are evicted when the cap is exceeded.  Defaults to
        ``4 * prompts_per_batch * rollouts_per_step``.
    """

    def __init__(
        self,
        prompts_per_batch: int,
        rollouts_per_step: int,
        stale_steps: int,
        rollout_worker: vLLMRollout,
        max_size: int | None = None,
    ) -> None:
        if prompts_per_batch <= 0:
            raise ValueError("prompts_per_batch must be > 0")
        if rollouts_per_step <= 0:
            raise ValueError("rollouts_per_step must be > 0")
        if stale_steps <= 0:
            raise ValueError("stale_steps must be > 0")

        self.prompts_per_batch = prompts_per_batch
        self.rollouts_per_step = rollouts_per_step
        self.stale_steps = stale_steps
        self.rollout_worker = rollout_worker

        self._batch_size: int = prompts_per_batch * rollouts_per_step
        _max = max_size if max_size is not None else 4 * self._batch_size
        self._store: deque[RolloutRecord] = deque(maxlen=_max)

        # Number of trainer update steps completed so far.
        self._trainer_steps: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def trainer_steps(self) -> int:
        return self._trainer_steps

    def __len__(self) -> int:
        return len(self._store)

    def add(self, rollouts: list[RolloutRecord], rewards: list[float]) -> None:
        """
        Ingest a batch of rollouts produced by the vLLM worker together with
        their scalar rewards.

        ``rollouts`` is the list returned by ``generate_rollouts`` /
        ``generate_rollouts_remote``.  ``rewards`` must have the same length.

        If the buffer would exceed ``max_size``, the oldest records are
        automatically evicted (``deque(maxlen=…)`` handles this).
        """
        if len(rollouts) != len(rewards):
            raise ValueError(
                f"rollouts ({len(rollouts)}) and rewards ({len(rewards)}) "
                "must have the same length"
            )
        for rollout, reward in zip(rollouts, rewards):
            response_record = rollout.response_record
            response_record.reward = [reward]
            self._store.append(rollout)
        logger.debug(
            f"Replay buffer: added {len(rollouts)} records, total {len(self._store)} / capacity {self._store.maxlen}."
        )

    def is_ready(self) -> bool:
        """Return True when at least one full trainer batch is available."""
        return len(self._store) >= self._batch_size

    def sample(self) -> dict[str, Any]:
        """
        Drain exactly one batch (``prompts_per_batch * rollouts_per_step``
        records) from the *front* of the buffer and return it as a dict
        consumable by ``prepare_batch``.

        Raises ``RuntimeError`` if the buffer does not yet hold a full batch.
        Use ``is_ready()`` before calling.
        """
        if not self.is_ready():
            raise RuntimeError(
                f"Buffer not ready: {len(self._store)} < {self._batch_size} records."
            )
        records: list[RolloutRecord] = [
            self._store.popleft() for _ in range(self._batch_size)
        ]
        rollouts = [
            {
                "prompt": r.prompt,
                "response": r.response,
                "prompt_ids": r.prompt_ids,
                "prompt_attention_mask": r.prompt_attention_mask,
                "response_ids": r.response_record.response_ids,
                "inference_logprobs": r.response_record.inference_logprobs,
                "metadata": r.metadata,
            }
            for r in records
        ]
        rewards = [r.response_record.reward for r in records]
        logger.debug(
            f"Replay buffer: sampled {self._batch_size} records, {len(self._store)} remaining."
        )
        return {"rollouts": rollouts, "rewards": rewards}

    def on_trainer_step(
        self,
        train_model: Any,
        model_update_group: Any,
        *,
        packed: bool = True,
        fsdp: bool = False,
        clear_kv_cache: bool = False,
    ) -> bool:
        """
        Call this once after each trainer gradient update.

        Increments the internal step counter and, every ``stale_steps``
        updates, pushes the trainer's current weights into the vLLM worker
        in-place via NCCL.

        Parameters
        ----------
        train_model:
            The (possibly DDP/FSDP-wrapped) trainer model.
        model_update_group:
            The NCCL process group shared between the trainer and the vLLM
            worker, returned by ``torch.distributed.new_group``.
        packed:
            Whether to send weights as a single packed tensor (faster).
        fsdp:
            Set to True when using FSDP so full tensors are gathered before
            sending.
        clear_kv_cache:
            If True, abort in-flight rollouts and clear the KV cache during
            the sync so every future token is generated under the new
            weights. If False, in-flight rollouts are frozen and resumed
            after the sync, keeping their (now stale) cache entries so no
            rollout work is lost.

        Returns
        -------
        bool
            True if a weight sync was performed this step.
        """
        self._trainer_steps += 1
        remaining = self.stale_steps - (self._trainer_steps % self.stale_steps)
        logger.info(
            f"[replay buffer] trainer step {self._trainer_steps} complete  "
            f"| buffer size: {len(self._store)}/{self._store.maxlen}  "
            f"| next sync in {remaining} step(s)"
        )
        if self._trainer_steps % self.stale_steps == 0:
            logger.info(
                f"[replay buffer] stale_steps={self.stale_steps} reached — triggering weight sync to vLLM"
            )
            self._sync_weights(
                train_model,
                model_update_group,
                packed=packed,
                fsdp=fsdp,
                clear_kv_cache=clear_kv_cache,
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sync_weights(
        self,
        train_model: Any,
        model_update_group: Any,
        *,
        packed: bool,
        fsdp: bool,
        clear_kv_cache: bool = False,
    ) -> None:
        logger.info(
            f"Syncing trainer weights to vLLM worker at step {self._trainer_steps} "
            f"(every {self.stale_steps} steps)."
        )
        asyncio.run(
            self.rollout_worker.update_weights(
                train_model,
                model_update_group,
                packed=packed,
                fsdp=fsdp,
                clear_kv_cache=clear_kv_cache,
            )
        )
        kv_cache_action = "aborted (cleared)" if clear_kv_cache else "kept (stale)"
        logger.info(f"Weight sync to vLLM complete. In-flight rollouts KV cache: {kv_cache_action}.")
