from typing import Any

import torch


def prepare_batch(
    rollouts: list[dict[str, Any]],
    rewards: list[float],
    tokenizer: Any,
    max_seq_len: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    input_ids_list, prompt_attn_list, response_mask_list = [], [], []
    prompt_lens, response_lens = [], []
    for rollout in rollouts:
        prompt_ids = rollout["prompt_ids"]
        prompt_attention_mask = rollout["prompt_attention_mask"]
        response_ids = rollout["response_ids"]
        full_ids = prompt_ids + response_ids
        # Response tokens are freshly generated (no internal padding), so
        # their attention mask is always 1; the prompt's mask comes from the
        # tokenizer, which may contain 0s.
        full_attn = prompt_attention_mask + [1] * len(response_ids)
        if len(full_ids) > max_seq_len:
            full_ids = full_ids[:max_seq_len]
            full_attn = full_attn[:max_seq_len]
            response_ids = full_ids[len(prompt_ids):]
        mask = [0] * len(prompt_ids) + [1] * len(response_ids)
        input_ids_list.append(full_ids)
        prompt_attn_list.append(full_attn)
        response_mask_list.append(mask)
        prompt_lens.append(len(prompt_ids))
        response_lens.append(len(response_ids))

    max_len = max(len(ids) for ids in input_ids_list)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    padded_ids = [ids + [pad_id] * (max_len - len(ids)) for ids in input_ids_list]
    padded_masks = [m + [0] * (max_len - len(m)) for m in response_mask_list]
    attn_masks = [a + [0] * (max_len - len(a)) for a in prompt_attn_list]

    inf_lp_list = []
    for rollout, P, R in zip(rollouts, prompt_lens, response_lens):
        row = [0.0] * (max_len - 1)
        for i, lp in enumerate(rollout["inference_logprobs"][:R]):
            row[P - 1 + i] = lp[0] if isinstance(lp, list) else lp
        inf_lp_list.append(row)

    return {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attn_masks, dtype=torch.long, device=device),
        "response_mask": torch.tensor(padded_masks, dtype=torch.float, device=device),
        "rewards": torch.tensor(rewards, dtype=torch.float, device=device),
        "inference_logprobs": torch.tensor(inf_lp_list, dtype=torch.float, device=device),
    }


def get_logprobs(model, input_ids, attention_mask, response_mask):
    """Compute per-response-token log-probs under `model`.

    Returns a tuple ``(token_logprobs, shift_mask)`` each of shape ``[B, T-1]``:
      - ``token_logprobs[b, t] = log π_θ(input_ids[b, t+1] | input_ids[b, :t+1])``
      - ``shift_mask[b, t] = 1.0`` iff ``input_ids[b, t+1]`` is a response token.

    Per-token (not per-sequence) log-probs are required so that PPO/DAPO/GRPO
    can apply per-token importance ratios and per-token clipping — the
    sample-level masked-mean form makes the clip bounds essentially non-
    functional (the geometric mean of many per-token ratios is always ≈ 1).
    """
    B, T = input_ids.shape
    shift_labels = input_ids[:, 1:]  # [B, T-1]
    shift_mask = response_mask[:, 1:]  # [B, T-1]
    token_logprobs = torch.zeros(B, T - 1, device=input_ids.device, dtype=torch.float32)

    # Chunk along the batch dimension to cap peak logit memory at
    # [chunk_size, T, V] instead of [B, T, V].
    chunk_size = 1024
    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits_chunk = model(
                input_ids=input_ids[start:end],
                attention_mask=attention_mask[start:end],
            ).logits  # [chunk, T, V]
        shift_logits = logits_chunk[:, :-1, :]  # [chunk, T-1, V]
        gathered = shift_logits.gather(
            -1, shift_labels[start:end].unsqueeze(-1)
        ).squeeze(
            -1
        )  # [chunk, T-1]
        token_logprobs[start:end] = (
            gathered - torch.logsumexp(shift_logits, dim=-1)
        ).float()
        del logits_chunk, shift_logits, gathered

    return token_logprobs, shift_mask
