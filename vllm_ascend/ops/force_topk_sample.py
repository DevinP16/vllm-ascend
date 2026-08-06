# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# force_topk sampler: small-operator reference implementation.
#
# This module provides the pre-fusion implementation assembled from standard
# torch operators. It is numerically equivalent to the future fused NPU kernel
# and serves as:
#   1. The initial implementation.
#   2. The regression golden for the future fused kernel.
#   3. The fallback path when the fused kernel is unavailable.
#
# Key invariants:
#   I1: greedy == full-vocab argmax (guaranteed by caller).
#   I2: top_k/top_p/min_p are exact when the effective
#       candidate set is a subset of top-k.
#   I3: raw logprobs use the full-vocab LSE (logsumexp),
#       matching raw_logprobs.
#
# Structure mirrors the private fork's fused_sample kernel:
#   Phase 1: topk (full-vocab -> [B, k])
#   Phase 2: raw logprobs (only when return_raw_logprobs=True)
#   Phase 3: _apply_sampling_constraints
#       (temperature + top_k + top_p + min_p + softmax)
#   Phase 4: _sample (Gumbel-max via exponential noise)
#   Phase 5: logprobs selection (raw vs processed)

import torch

from vllm_ascend.sample.topk_map import CompactDist

__all__ = ["build_compact_for_logprobs", "force_topk_sample"]

# Larger compiled softmax kernels overflow 910B unified-buffer capacity.
_MAX_COMPILED_TOPK = 4096


def build_compact_for_logprobs(
    logits: torch.Tensor, k: int
) -> CompactDist:
    """Build a CompactDist for logprobs reporting (no sampling).

    Used by the greedy branch of AscendSampler.sample() when the caller
    only needs the compact logprob representation, not a random sample.

    Args:
        logits: [B, V] float32, post-logits-processors, pre-temperature.
        k: global candidate ceiling.

    Returns:
        CompactDist with token_index [B, k] i32 and logprobs [B, k] f32.
    """
    k = min(k, logits.shape[-1])
    lse_full = torch.logsumexp(logits, dim=-1, keepdim=True)  # [B, 1]
    topv, token_index = torch.topk(logits, k, dim=-1)  # [B, k] descending
    logprobs = topv - lse_full  # [B, k]
    return CompactDist(token_index.to(torch.int32), logprobs)


def _apply_sampling_constraints(
    topv: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: torch.Tensor,
    min_p: torch.Tensor | None,
    k: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply temperature + top_k + top_p + min_p masks + softmax.

    Args:
        topv: [B, k] float32, top-k logits (descending).
        temperature: [B] float32, per-request.
        top_p: [B] float32, per-request (1.0 = disabled).
        top_k: [B] int32, per-request (<=0 = disabled -> use k).
        min_p: [B] float32 or None, per-request.
        k: candidate ceiling.
        device: NPU device.

    Returns:
        s_masked: [B, k] float32, masked logits after temperature +
            top_k/top_p/min_p.
        probs_k: [B, k] float32, softmax probabilities (renormalized).
    """
    # Temperature scaling
    s = topv / temperature.unsqueeze(1)  # [B, k]

    neg = float("-inf")

    # min_p is applied after temperature and before top_k/top_p in vLLM.
    # Normalizing over k does not change the probability ratio to top-1.
    if min_p is not None:
        min_p_probs = torch.softmax(s, dim=-1)
        threshold = min_p[:, None] * min_p_probs[:, :1]
        s = s.masked_fill(min_p_probs < threshold, neg)

    # top_k: mask ranks >= min(top_k, k). Already descending, just cut.
    k_cap = torch.where(
        top_k > 0,
        torch.minimum(top_k, torch.full_like(top_k, k)),
        torch.full_like(top_k, k),
    )  # [B] int32
    rank = torch.arange(k, device=device, dtype=top_k.dtype)  # [k]
    s = s.masked_fill(rank[None, :] >= k_cap[:, None], neg)  # [B, k]

    # vLLM applies top_p after top_k and normalizes over the remaining logits.
    top_p_probs = torch.softmax(s, dim=-1)
    # keep[r] = (cumulative prob BEFORE r) < top_p,
    # includes threshold-crossing item.
    cdf = top_p_probs.cumsum(dim=-1)  # [B, k]
    keep = (cdf - top_p_probs) < top_p[:, None]  # [B, k]
    s = s.masked_fill(~keep, neg)  # [B, k]

    # Softmax over k candidates (renormalization after masking)
    probs_k = torch.softmax(s, dim=-1)  # [B, k]
    return s, probs_k


def _sample(
    probs_k: torch.Tensor,
    generators: dict[int, torch.Generator],
    B: int,
    token_index: torch.Tensor,
) -> torch.Tensor:
    """Gumbel-max sampling via exponential noise.

    No CPU-NPU sync (design N1). Aligned with
    vllm_ascend/sample/sampler.py::random_sample.

    Args:
        probs_k: [B, k] float32, renormalized probabilities.
        generators: per-request torch.Generator dict (may be empty).
        B: batch size.
        token_index: [B, k] int64, vocab id mapping (for pi mapping).

    Returns:
        [B] int64, sampled vocab ids.
    """
    q = torch.empty_like(probs_k)
    if len(generators) != B:
        q.exponential_()
    if generators:
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)

    local = (probs_k / q).argmax(dim=-1)  # [B] local rank
    sampled = token_index.gather(
        1, local[:, None]
    ).squeeze(1).to(torch.int64)  # [B]
    return sampled


def _force_topk_tensors(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: torch.Tensor,
    min_p: torch.Tensor | None,
    lse_full: torch.Tensor | None,
    k: int,
    return_raw_logprobs: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the compact distribution using tensor-only operations."""
    _, V = logits.shape
    k = min(k, V)

    topv, token_index = torch.topk(logits, k, dim=-1)  # [B, k] desc

    s_masked, probs_k = _apply_sampling_constraints(
        topv, temperature, top_p, top_k, min_p,
        k, logits.device,
    )

    if return_raw_logprobs:
        assert lse_full is not None
        logprobs = topv - lse_full
    else:
        logprobs = torch.log_softmax(s_masked, dim=-1)
    return probs_k, token_index, logprobs


_compiled_force_topk_tensors = torch.compile(
    _force_topk_tensors,
    dynamic=False,
    options={"npu_backend": "ascendc"},
)


def force_topk_sample(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: torch.Tensor,
    min_p: torch.Tensor | None,
    generators: dict[int, torch.Generator],
    k: int,
    return_raw_logprobs: bool = True,
) -> tuple[torch.Tensor, CompactDist]:
    """Sample from logits using the force_topk compact-space path.

    Tensor-only candidate processing is compiled. Generator-aware random
    sampling remains eager because torch.Generator cannot be represented in
    the compiled graph.
    """
    B, V = logits.shape
    k = min(k, V)
    if return_raw_logprobs:
        # Dynamic full-vocab reductions are not supported by the Ascend
        # Triton backend used by this image, so keep this reduction eager.
        lse_full = torch.logsumexp(logits, dim=-1, keepdim=True)
    else:
        lse_full = None

    tensor_path = (
        _force_topk_tensors
        if k > _MAX_COMPILED_TOPK
        else _compiled_force_topk_tensors
    )
    probs_k, token_index, logprobs = tensor_path(
        logits,
        temperature,
        top_p,
        top_k,
        min_p,
        lse_full,
        k,
        return_raw_logprobs,
    )
    sampled = _sample(probs_k, generators, B, token_index)
    return sampled, CompactDist(token_index.to(torch.int32), logprobs)
