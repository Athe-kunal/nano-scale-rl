import asyncio
import time
import urllib.request
from typing import Any

import aiohttp
import torch
from transformers import PreTrainedTokenizerBase
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
) -> list[int]:
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(  # type: ignore[return-value]
        messages, add_generation_prompt=True, tokenize=True
    )


class vLLMRollout:
    def __init__(self, base_url: str, tokenizer: PreTrainedTokenizerBase):
        # vLLM is already running as a server at base_url
        # started with VLLM_SERVER_DEV_MODE=1
        self.base_url = base_url
        self.tokenizer = tokenizer
        self.session = aiohttp.ClientSession()

    async def initialize_weight_transfer(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
    ) -> None:
        async with self.session.post(
            f"{self.base_url}/init_weight_transfer",
            json=dict(
                master_address=master_address,
                master_port=master_port,
                rank_offset=rank_offset,
                world_size=world_size,
            ),
        ) as resp:
            resp.raise_for_status()

    async def update_weights(
        self,
        train_model: Any,
        model_update_group: Any,
        *,
        packed: bool = True,
        fsdp: bool = False,
    ) -> None:
        if hasattr(train_model, "module"):
            train_model = train_model.module

        names, dtype_names, shapes = collect_weight_metadata(train_model, fsdp=fsdp)

        async with self.session.post(
            f"{self.base_url}/pause", json={"mode": "keep"}
        ) as resp:
            resp.raise_for_status()

        try:
            async with self.session.post(
                f"{self.base_url}/start_weight_update",
                json=dict(names=names, dtype_names=dtype_names, shapes=shapes, packed=packed),
            ) as resp:
                resp.raise_for_status()

            NCCLWeightTransferEngine.trainer_send_weights(
                _iter_model_parameters(train_model, fsdp=fsdp),
                NCCLTrainerSendWeightsArgs(group=model_update_group, packed=packed),
            )

            async with self.session.post(f"{self.base_url}/finish_weight_update") as resp:
                resp.raise_for_status()
        finally:
            async with self.session.post(f"{self.base_url}/resume") as resp:
                resp.raise_for_status()

    async def generate(
        self,
        prompt: str,
        sampling_params: dict,
        system_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RolloutRecord:
        prompt_ids = apply_chat_template(self.tokenizer, prompt, system_prompt)

        # standard OpenAI-compatible endpoint
        async with self.session.post(
            f"{self.base_url}/v1/completions",
            json={"prompt": prompt, **sampling_params},
        ) as resp:
            data = await resp.json()

        choice = data["choices"][0]
        response_text = choice["text"]
        token_ids: list[int] = choice["logprobs"]["tokens"]
        token_logprobs: list[float] = choice["logprobs"]["token_logprobs"]

        response_record = ResponseRecord(
            response_ids=token_ids,
            inference_logprobs=[[lp] for lp in token_logprobs],
            reward=[],
        )
        return RolloutRecord(
            prompt=prompt,
            response=response_text,
            prompt_ids=prompt_ids,
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
        """
        tasks = [
            self.generate(prompt, sampling_params, system_prompt)
            for prompt in prompts
            for _ in range(num_samples)
        ]
        return list(await asyncio.gather(*tasks))


def wait_for_rollout_worker(base_url: str, timeout_s: int = 300) -> None:
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{base_url.rstrip('/')}/health")
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = resp.read()
            if body:
                import json
                payload = json.loads(body)
                if payload.get("ok"):
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
