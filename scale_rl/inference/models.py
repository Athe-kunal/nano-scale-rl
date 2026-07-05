from dataclasses import dataclass
from typing import Any

@dataclass
class ResponseRecord:
    response_ids: list[int]
    inference_logprobs: list[list[float]]
    reward: list[float]
    finish_reason: str = ""  # "stop" (hit EOS/stop string) or "length" (hit max_tokens)

@dataclass
class RolloutRecord:
    """One (prompt, completion) pair with its reward and inference log-probs."""
    prompt: str
    response: str
    prompt_ids: list[int]
    prompt_attention_mask: list[int]
    response_record: ResponseRecord
    metadata: dict[str, Any]