from types import SimpleNamespace

import torch
from vllm.v1.sample.logits_processor.state import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata

import vllm_ascend.sample.sampler as sampler_module
from vllm_ascend.ops.force_topk_sample import force_topk_sample
from vllm_ascend.sample.sampler import AscendSampler

VOCAB_SIZE = 151_936
BATCH_SIZES = (8, 32, 1)
FORCE_TOPKS = (128, 512, 2048, 8192)
LOGPROBS_MODES = ("raw_logprobs", "processed_logprobs")


def _make_logits(batch_size: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260806 + batch_size)
    logits = torch.randn(batch_size, VOCAB_SIZE, generator=generator, dtype=torch.float32)
    return (logits * 3.0).npu()


def _make_inputs(batch_size: int):
    device = torch.device("npu")
    temperature = torch.linspace(0.7, 1.3, batch_size, device=device)
    top_p = torch.full((batch_size,), 0.9, device=device)
    top_k = torch.full((batch_size,), 64, dtype=torch.int32, device=device)
    min_p = torch.full((batch_size,), 0.02, device=device)
    return temperature, top_p, top_k, min_p


def _make_generators(batch_size: int, seed: int) -> dict[int, torch.Generator]:
    return {index: torch.Generator(device="npu").manual_seed(seed + index) for index in range(batch_size)}


def _assert_sample_in_reference_candidates(
    logits: torch.Tensor,
    sampled: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: torch.Tensor,
    min_p: torch.Tensor,
) -> None:
    scaled = logits / temperature.unsqueeze(1)

    # Min-p only depends on the probability ratio to the row maximum, so the
    # normalization term cancels and need not be materialized.
    relative_probs = torch.exp(scaled - scaled.max(dim=-1, keepdim=True).values)
    scaled = scaled.masked_fill(relative_probs < min_p.unsqueeze(1), float("-inf"))

    max_top_k = int(top_k.max().cpu())
    values, indices = scaled.topk(max_top_k, dim=-1)
    ranks = torch.arange(max_top_k, device=logits.device)
    top_k_keep = ranks.unsqueeze(0) < top_k.unsqueeze(1)
    values = values.masked_fill(~top_k_keep, float("-inf"))

    probabilities = values.softmax(dim=-1)
    cumulative = probabilities.cumsum(dim=-1)
    top_p_keep = cumulative - probabilities < top_p.unsqueeze(1)
    candidates = top_k_keep & top_p_keep
    sampled_is_valid = ((indices == sampled.unsqueeze(1)) & candidates).any(dim=-1)
    assert sampled_is_valid.all()


def _make_metadata(batch_size: int, max_num_logprobs: int):
    device = torch.device("npu")
    temperature, top_p, top_k, _ = _make_inputs(batch_size)
    return SamplingMetadata(
        temperature=temperature,
        all_greedy=False,
        all_random=True,
        top_p=top_p,
        top_k=top_k,
        generators={},
        max_num_logprobs=max_num_logprobs,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(batch_size, device=device),
        presence_penalties=torch.zeros(batch_size, device=device),
        repetition_penalties=torch.ones(batch_size, device=device),
        output_token_ids=[[] for _ in range(batch_size)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
        logprob_token_ids={},
    )


def test_force_topk_npu_matrix():
    for logprobs_mode in LOGPROBS_MODES:
        return_raw_logprobs = logprobs_mode == "raw_logprobs"
        for force_topk in FORCE_TOPKS:
            for batch_size in BATCH_SIZES:
                logits = _make_logits(batch_size)
                temperature, top_p, top_k, min_p = _make_inputs(batch_size)
                sampled, compact = force_topk_sample(
                    logits,
                    temperature,
                    top_p,
                    top_k,
                    min_p,
                    {},
                    force_topk,
                    return_raw_logprobs=return_raw_logprobs,
                )

                assert sampled.dtype == torch.int64
                assert compact.token_index.dtype == torch.int32
                assert not torch.isnan(compact.logprobs).any()
                _assert_sample_in_reference_candidates(
                    logits,
                    sampled,
                    temperature,
                    top_p,
                    top_k,
                    min_p,
                )

                if return_raw_logprobs:
                    full = torch.log_softmax(logits, dim=-1)
                    expected = full.gather(1, compact.token_index.to(torch.int64))
                    max_error = (compact.logprobs - expected).abs().max()
                    assert max_error <= 1e-4, f"batch={batch_size}, k={force_topk}, max_error={max_error}"

                torch.npu.synchronize()
                torch.npu.empty_cache()


def test_force_topk_npu_seed_reproducible():
    batch_size = 8
    logits = _make_logits(batch_size)
    temperature, top_p, top_k, min_p = _make_inputs(batch_size)

    first, _ = force_topk_sample(
        logits,
        temperature,
        top_p,
        top_k,
        min_p,
        _make_generators(batch_size, 1234),
        512,
    )
    second, _ = force_topk_sample(
        logits,
        temperature,
        top_p,
        top_k,
        min_p,
        _make_generators(batch_size, 1234),
        512,
    )
    assert torch.equal(first, second)


def test_force_topk_greedy_and_logprobs_fallback(monkeypatch):
    monkeypatch.setattr(
        sampler_module,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_reduce_sample=False),
    )
    sampler = AscendSampler(logprobs_mode="raw_logprobs")
    sampler.force_topk = 128

    logits = _make_logits(8)
    greedy = sampler.greedy_sample(logits)
    assert torch.equal(greedy, logits.argmax(dim=-1))

    metadata = _make_metadata(batch_size=8, max_num_logprobs=129)
    assert not sampler._force_topk_enabled(metadata)
