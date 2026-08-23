"""Shared DSV4 utility functions used across BF16 and FP8 paths."""

import os
from types import MethodType

import torch
from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.models_py.modules.factory.linear import LinearFactory

_V4_FP8_BLOCK_CFG = Fp8BlockWiseQuantConfig()


def _sm120_forward_quantized_cached_scale(
    self, input_fp8: torch.Tensor, input_scales: torch.Tensor, output: torch.Tensor
) -> torch.Tensor:
    """DSV4 SM120 FP8 GEMM with load-time decoded weight scales."""
    from flashinfer.gemm import gemm_fp8_nt_groupwise

    if input_fp8.size(0) == 0:
        return output
    a_scale = self._ue8m0_packed_to_fp32(
        input_scales, (self.K + 127) // 128
    )
    gemm_fp8_nt_groupwise(
        input_fp8.contiguous(),
        self.weight,
        a_scale,
        self._dsv4_sm120_weight_scale_fp32,
        scale_granularity_mnk=(1, 128, 128),
        scale_major_mode="K",
        out=output,
        out_dtype=torch.bfloat16,
    )
    if self.bias is not None:
        output.add_(self.bias.to(output.dtype))
    return output


def _enable_sm120_cached_weight_scale(linear):
    """Specialize a DSV4 FP8 linear without changing the factory class."""
    weight = getattr(linear, "weight", None)
    if (
        os.environ.get("DSV4_SM120_CACHE_FP8_SCALES", "1") == "0"
        or weight is None
        or not weight.is_cuda
        or torch.cuda.get_device_capability(weight.device)[0] != 12
    ):
        return linear
    weight_scale = linear._ue8m0_packed_to_fp32(
        linear.weight_scales, (linear.K + 127) // 128
    )
    if weight_scale.size(0) == linear.N:
        weight_scale = weight_scale[::128]
    linear._dsv4_sm120_weight_scale_fp32 = weight_scale.contiguous()
    linear._forward_quantized_sm120 = MethodType(
        _sm120_forward_quantized_cached_scale, linear
    )
    return linear


def _repack_v4_fp8_scale_to_int32(scale: torch.Tensor) -> torch.Tensor:
    """V4 ckpt UE8M0 ``[N/128, K/128]`` to DeepGEMM int32-packed scale."""
    assert scale.dtype == torch.float8_e8m0fnu, f"unexpected scale dtype {scale.dtype}"
    assert scale.dim() == 2, f"unexpected scale dim {scale.dim()}"

    n_blk, _ = scale.shape
    n = n_blk * 128
    idx = torch.arange(n, device=scale.device) // 128
    scale_rep = scale.float().index_select(-2, idx)
    return get_mn_major_tma_aligned_packed_ue8m0_tensor(scale_rep)


def _v4_fp8_linear(w: torch.Tensor, s: torch.Tensor):
    """Build a CudaFp8DeepGEMMLinear from raw V4 FP8 weight + scale tensors."""
    assert s is not None, "expected non-null FP8 scale"
    if s.dtype == torch.float8_e8m0fnu:
        s = _repack_v4_fp8_scale_to_int32(s)
    local = {"_w": w, "_s": s}
    linear = LinearFactory.create_linear_from_weights(
        local,
        "_w",
        "_s",
        quant_config=_V4_FP8_BLOCK_CFG,
    )
    return _enable_sm120_cached_weight_scale(linear)


def _v4_fp8_linear_from_dict(weights: dict, weight_key: str, scale_key: str):
    """Backwards-compat bridge over ``_v4_fp8_linear`` for flat dict callers."""
    w = weights[weight_key]
    s = weights[scale_key]
    if s.dtype == torch.float8_e8m0fnu:
        s = _repack_v4_fp8_scale_to_int32(s)
        weights[scale_key] = s
    return _v4_fp8_linear(w, s)


def _sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Reference PyTorch sparse attention with attention sink.

    Output: [B, S, H, D]
    """
    bsz, seqlen, n_heads, head_dim = q.size()
    valid = topk_idxs >= 0
    safe_idxs = topk_idxs.clamp_min(0)

    idx_expanded = safe_idxs.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    kv_exp = kv.unsqueeze(1).expand(-1, seqlen, -1, -1)
    selected = torch.gather(kv_exp, 2, idx_expanded)

    q_f = q.float()
    selected_f = selected.float()
    logits = torch.einsum("bshd,bskd->bshk", q_f, selected_f) * softmax_scale
    logits = logits.masked_fill(~valid.unsqueeze(2), float("-inf"))

    scores_max = logits.amax(dim=-1, keepdim=True).clamp_min(-1e30)
    exp_logits = torch.exp(logits - scores_max)
    sink_logit = sink.view(1, 1, n_heads, 1).expand_as(scores_max)
    exp_sink = torch.exp(sink_logit - scores_max)
    sum_exp = exp_logits.sum(dim=-1, keepdim=True) + exp_sink

    acc_o = torch.einsum("bshk,bskd->bshd", exp_logits, selected_f)
    out = acc_o / sum_exp
    return out.to(q.dtype)
