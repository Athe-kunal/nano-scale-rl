from dataclasses import dataclass
from typing import Any

@dataclass
class ResponseRecord:
    response_ids: list[int]
    inference_logprobs: list[list[float]]
    reward: list[float]

@dataclass
class RolloutRecord:
    """One (prompt, completion) pair with its reward and inference log-probs."""
    prompt: str
    response: str
    prompt_ids: list[int]
    response_record: ResponseRecord
    metadata: dict[str, Any]