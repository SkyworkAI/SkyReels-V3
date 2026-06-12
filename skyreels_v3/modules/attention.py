# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    from flash_attn import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

__all__ = [
    "flash_attention",
    "attention",
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == "cuda" and q.size(-1) <= 256

    # params
    b, lq, lk, _ = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query

    q = half(q.flatten(0, 1))
    q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(
        device=q.device, non_blocking=True
    )

    # preprocess key, value

    k = half(k.flatten(0, 1))
    v = half(v.flatten(0, 1))
    k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(
        device=k.device, non_blocking=True
    )

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            "Flash attention 3 is not available, use flash attention 2 instead."
        )

    torch.cuda.nvtx.range_push(
        f"{list(q.shape)}-{list(k.shape)}-{list(v.shape)}-{q.dtype}-{k.dtype}-{v.dtype}"
    )
    # apply attention
    # NOTE: flash-attn >= 2.5.9 changed the default return type from Tuple to Tensor.
    # Use _extract_output() to handle both old and new API.
    def _extract_output(result):
        """兼容处理: tuple 取第一个元素, tensor 直接返回"""
        if isinstance(result, tuple):
            return result[0]
        return result

    fa_success = False
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        try:
            result = flash_attn_interface.flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
                .cumsum(0, dtype=torch.int32)
                .to(q.device, non_blocking=True),
                cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
                .cumsum(0, dtype=torch.int32)
                .to(q.device, non_blocking=True),
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic,
            )
            x = _extract_output(result).unflatten(0, (b, lq))
            fa_success = True
        except Exception as e:
            warnings.warn(f"FlashAttention 3 failed: {e}. Falling back to FlashAttention 2 / PyTorch.")

    if not fa_success and FLASH_ATTN_2_AVAILABLE:
        try:
            result = flash_attn.flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
                .cumsum(0, dtype=torch.int32)
                .to(q.device, non_blocking=True),
                cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
                .cumsum(0, dtype=torch.int32)
                .to(q.device, non_blocking=True),
                max_seqlen_q=lq,
                max_seqlen_k=lk,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
            )
            x = _extract_output(result).unflatten(0, (b, lq))
            fa_success = True
        except Exception as e:
            warnings.warn(f"FlashAttention 2 failed: {e}. Falling back to PyTorch native attention.")

    if not fa_success:
        # Fall back to PyTorch native scaled_dot_product_attention
        warnings.warn("Using PyTorch native scaled_dot_product_attention as fallback.")
        q_orig = q.unflatten(0, (b, lq)).transpose(1, 2)
        k_orig = k.unflatten(0, (b, lk)).transpose(1, 2)
        v_orig = v.unflatten(0, (b, lk)).transpose(1, 2)
        x = torch.nn.functional.scaled_dot_product_attention(
            q_orig, k_orig, v_orig, attn_mask=None, is_causal=causal, dropout_p=dropout_p
        ).transpose(1, 2).contiguous()

    torch.cuda.nvtx.range_pop()

    # output
    return x


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                "Padding mask is disabled when using scaled_dot_product_attention. "
                "It can have a significant impact on performance."
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p
        )

        out = out.transpose(1, 2).contiguous()
        return out
