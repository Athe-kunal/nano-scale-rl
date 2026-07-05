import asyncio
import time
import urllib.request
from typing import Any

import aiohttp
import torch
from tqdm import tqdm
from transformers import BatchEncoding, PreTrainedTokenizerBase
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
)

from scale_rl.inference.models import ResponseRecord, RolloutRecord


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _iter_model_parameters(model: Any, fsdp: bool):
    for name, param in model.named_parameters():
        if fsdp and hasattr(param, "full_tensor"):
            yield name, param.full_tensor()
        else:
            yield name, param


def collect_weight_metadata(model: Any, fsdp: bool = False) -> tuple[list[str], list[str], list[list[int]]]:
    names, dtype_names, shapes = [], [], []
    for name, param in _iter_model_parameters(model, fsdp=fsdp):
        names.append(name)
        dtype_names.append(_dtype_name(param.dtype))
        shapes.append(list(param.shape))
    return names, dtype_names, shapes


def apply_chat_template(
    tokenizer: PreTrainedTokenizerBase, prompt: str, system_prompt: str | None = None
) -> tuple[list[int], list[int]]:
    """Returns (prompt_ids, attention_mask)."""
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    encoded: BatchEncoding = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    return encoded["input_ids"], encoded["attention_mask"]


class vLLMRollout:
    def __init__(self, base_url: str, tokenizer: PreTrainedTokenizerBase):
        # vLLM is already running as a server at base_url
        # started with VLLM_SERVER_DEV_MODE=1
        self.base_url = base_url
        self.tokenizer = tokenizer

    async def initialize_weight_transfer(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
    ) -> None:
        # Each public method here is invoked from its own separate
        # asyncio.run(...) call at the call site (a fresh event loop every
        # time), so the ClientSession must be opened and closed within this
        # call rather than cached on self — an aiohttp session is bound to
        # the loop that was running when it was created.
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/init_weight_transfer_engine",
                json={
                    "init_info": dict(
                        master_address=master_address,
                        master_port=master_port,
                        rank_offset=rank_offset,
                        world_size=world_size,
                    )
                },
            ) as resp:
                resp.raise_for_status()

    async def update_weights(
        self,
        train_model: Any,
        model_update_group: Any,
        *,
        packed: bool = True,
        fsdp: bool = False,
        clear_kv_cache: bool = False,
    ) -> None:
        if hasattr(train_model, "module"):
            train_model = train_model.module

        names, dtype_names, shapes = collect_weight_metadata(train_model, fsdp=fsdp)

        # mode="abort" drops in-flight rollouts and clears the KV cache, so
        # every future token is generated under the new weights (correct but
        # wastes partial rollouts). mode="keep" freezes in-flight requests
        # and resumes them after the update, keeping their (now stale) cache
        # entries so no rollout work is lost.
        pause_mode = "abort" if clear_kv_cache else "keep"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/pause",
                params={"mode": pause_mode, "clear_cache": str(clear_kv_cache).lower()},
            ) as resp:
                resp.raise_for_status()

            try:
                # The server's /update_weights handler blocks until it has
                # received the matching NCCL broadcast (it calls the same
                # collective op on the vLLM worker side), so the HTTP request
                # and the trainer's (blocking, synchronous) NCCL send must run
                # concurrently rather than one-after-the-other.
                async def _post_update_weights() -> None:
                    async with session.post(
                        f"{self.base_url}/update_weights",
                        json={
                            "update_info": dict(
                                names=names,
                                dtype_names=dtype_names,
                                shapes=shapes,
                                packed=packed,
                            )
                        },
                    ) as resp:
                        resp.raise_for_status()

                await asyncio.gather(
                    _post_update_weights(),
                    asyncio.to_thread(
                        NCCLWeightTransferEngine.trainer_send_weights,
                        _iter_model_parameters(train_model, fsdp=fsdp),
                        NCCLTrainerSendWeightsArgs(group=model_update_group, packed=packed),
                    ),
                )
            finally:
                async with session.post(f"{self.base_url}/resume") as resp:
                    resp.raise_for_status()

    async def generate(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        sampling_params: dict,
        system_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RolloutRecord:
        prompt_ids, prompt_attention_mask = apply_chat_template(
            self.tokenizer, prompt, system_prompt
        )

        # standard OpenAI-compatible endpoint. return_token_ids=True is a vLLM
        # extension that adds an integer `token_ids` field to the response —
        # logprobs.tokens is always list[str] (human-readable token strings,
        # e.g. "Ġhello"), never usable as ids for training.
        async with session.post(
            f"{self.base_url}/v1/completions",
            json={"prompt": prompt, "return_token_ids": True, **sampling_params},
        ) as resp:
            data = await resp.json()

        choice = data["choices"][0]
        response_text = choice["text"]
        token_ids: list[int] = choice["token_ids"]
        token_logprobs: list[float] = choice["logprobs"]["token_logprobs"]

        response_record = ResponseRecord(
            response_ids=token_ids,
            inference_logprobs=[[lp] for lp in token_logprobs],
            reward=[],
            finish_reason=choice.get("finish_reason", ""),
        )
        return RolloutRecord(
            prompt=prompt,
            response=response_text,
            prompt_ids=prompt_ids,
            prompt_attention_mask=prompt_attention_mask,
            response_record=response_record,
            metadata=metadata or {},
        )


    async def generate_batch(
        self,
        prompts: list[str],
        num_samples: int,
        sampling_params: dict,
        system_prompt: str | None = None,
    ) -> list[RolloutRecord]:
        """Generate `num_samples` completions per prompt concurrently.

        Returns a flat list ordered as [prompt_0 × num_samples, prompt_1 × num_samples, ...].
        The progress bar advances by one prompt once all of its samples land
        (samples for different prompts can complete out of order).
        """
        async def _generate_indexed(
            session: aiohttp.ClientSession, flat_idx: int, prompt_idx: int, prompt: str
        ) -> tuple[int, int, RolloutRecord]:
            record = await self.generate(session, prompt, sampling_params, system_prompt)
            return flat_idx, prompt_idx, record

        # A single session is shared across all concurrent requests in this
        # batch (all running under the one event loop that asyncio.run(...)
        # created for this call), so TCP connections get reused.
        async with aiohttp.ClientSession() as session:
            tasks = [
                _generate_indexed(session, flat_idx, prompt_idx, prompt)
                for prompt_idx, prompt in enumerate(prompts)
                for flat_idx in range(prompt_idx * num_samples, (prompt_idx + 1) * num_samples)
            ]
            results: list[RolloutRecord | None] = [None] * len(tasks)
            landed_per_prompt = [0] * len(prompts)
            with tqdm(total=len(prompts), desc="generating rollouts") as pbar:
                for coro in asyncio.as_completed(tasks):
                    flat_idx, prompt_idx, record = await coro
                    results[flat_idx] = record
                    landed_per_prompt[prompt_idx] += 1
                    if landed_per_prompt[prompt_idx] == num_samples:
                        pbar.update(1)
            return results


def wait_for_rollout_worker(base_url: str, timeout_s: int = 300) -> None:
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{base_url.rstrip('/')}/health")
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    return
        except Exception as e:
            last_err = e
        time.sleep(1.0)
    raise RuntimeError(
        f"Rollout worker at {base_url} did not become healthy within {timeout_s}s"
        + (f"; last error: {last_err}" if last_err else "")
    )



def initialize_trainer(
    master_address: str, master_port: int, world_size: int
) -> tuple[NCCLTrainerSendWeightsArgs, PyNcclCommunicator]:
    trainer_group = NCCLWeightTransferEngine.trainer_init(
        dict(
            master_address=master_address,
            master_port=master_port,
            world_size=world_size,
        )
    )
    trainer_args = NCCLTrainerSendWeightsArgs(
        group=trainer_group,
        packed=True,  # use packed broadcasting for efficiency
    )
    return trainer_args, trainer_group
