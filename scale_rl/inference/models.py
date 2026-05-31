from dataclasses import dataclass


@dataclass
class RolloutRecord:
    """One (prompt, completion) pair with its reward and inference log-probs."""
    prompt: str
    response: str
    prompt_ids: list[int]
    response_ids: list[int]
    inference_logprobs: list[float]
    reward: float
