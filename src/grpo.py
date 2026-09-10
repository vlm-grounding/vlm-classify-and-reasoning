# ============================================================
# GRPO math: group advantages, completion log-probs, policy loss.
# ============================================================

from __future__ import annotations

import torch
import torch.nn.functional as F


def group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Standardize rewards inside one prompt group. Shape [G]."""
    if rewards.ndim != 1:
        raise ValueError(f"rewards must be 1-D [G], got {tuple(rewards.shape)}")
    if rewards.numel() == 1:
        return torch.zeros_like(rewards)
    centered = rewards - rewards.mean()
    return centered / (rewards.std(unbiased=False) + eps)


def completion_token_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_len: int,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-token log π of completion tokens.

    logits:     [B, T, V]  (predicts next token)
    input_ids:  [B, T]
    prompt_len: first completion token is at index prompt_len
    returns (token_logp [B, T-1], mask [B, T-1])
    """
    logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    target = input_ids[:, 1:]
    token_logp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    idx = torch.arange(token_logp.size(1), device=token_logp.device)
    mask = (idx >= max(prompt_len - 1, 0)).to(dtype=token_logp.dtype)
    mask = mask.unsqueeze(0).expand_as(token_logp)
    if attention_mask is not None:
        mask = mask * attention_mask[:, 1:].to(dtype=token_logp.dtype)
    return token_logp, mask


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp(min=1.0)
    return (values * mask).sum() / denom


def sequence_mean_logprob(token_logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean log-prob per sequence. token_logp/mask: [B, T]. returns [B]."""
    denom = mask.sum(dim=-1).clamp(min=1.0)
    return (token_logp * mask).sum(dim=-1) / denom


def grpo_loss(
    token_logp: torch.Tensor,
    token_logp_ref: torch.Tensor,
    mask: torch.Tensor,
    advantages: torch.Tensor,
    kl_beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    L = -mean_i( Â_i * mean_t log π(o_i,t) ) + β * mean KL(π || π_ref)

    advantages: [B]  (detached group-normalized rewards)
    """
    seq_logp = sequence_mean_logprob(token_logp, mask)
    seq_logp_ref = sequence_mean_logprob(token_logp_ref, mask)
    policy = -(advantages.detach() * seq_logp).mean()
    kl = masked_mean(token_logp - token_logp_ref.detach(), mask)
    loss = policy + float(kl_beta) * kl
    stats = {
        "loss_policy": float(policy.detach()),
        "kl": float(kl.detach()),
        "seq_logp": float(seq_logp.mean().detach()),
    }
    return loss, stats
