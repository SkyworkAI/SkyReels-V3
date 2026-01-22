# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
from xfuser.core.long_ctx_attention import xFuserLongContextAttention

from ..modules.transformer import sinusoidal_embedding_1d
from ..utils.avatar_util import get_attn_map_with_target


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(pad_size, s1, s2, dtype=original_tensor.dtype, device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    output = []
    for i, freqs_i_rank in enumerate(freqs):

        # precompute multipliers
        x_i = x[i].to(torch.float64).unflatten(-1, (-1, 2))  # .reshape(s, n, -1, 2)
        x_i_sin = x_i[..., 0] * freqs_i_rank[..., 0] - x_i[..., 1] * freqs_i_rank[..., 1]
        x_i_cos = x_i[..., 0] * freqs_i_rank[..., 1] + x_i[..., 1] * freqs_i_rank[..., 0]
        x_i = torch.stack([x_i_sin, x_i_cos], dim=-1).flatten(2)

        # append to collection
        output.append(x_i)
    return torch.stack(output)


def usp_dit_forward_avatar(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
    audio=None,
    ref_target_masks=None,
    audio_mask=None,
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """

    assert clip_fea is not None and y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    _, T, H, W = x[0].shape
    N_t = T // self.patch_size[0]
    N_h = H // self.patch_size[1]
    N_w = W // self.patch_size[2]

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
    x[0] = x[0].to(context[0].dtype)

    # time embeddings
    # with amp.autocast(dtype=torch.float32):
    e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).to(self.patch_embedding.weight.dtype))
    e0 = self.time_projection(e).unflatten(1, (6, self.dim))
    # assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

    # build rope freqs
    sp_size = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()
    s = x.size(1) // sp_size
    c = self.dim // self.num_heads // 2
    # split freqs
    freqs = self.freqs.cuda().split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    # loop over samples
    freqs_l = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        # precompute multipliers
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        freqs_i_rank = freqs_i[(sp_rank * s) : ((sp_rank + 1) * s), :, :]
        # if self.enable_compile:
        freqs_i_rank = torch.view_as_real(freqs_i_rank)
        freqs_l.append(freqs_i_rank)
    freqs = freqs_l

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
    )

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1)

    # get audio token
    audio_cond = audio.to(device=x.device, dtype=x.dtype)
    first_frame_audio_emb_s = audio_cond[:, :1, ...]
    latter_frame_audio_emb = audio_cond[:, 1:, ...]
    latter_frame_audio_emb = rearrange(latter_frame_audio_emb, "b (n_t n) w s c -> b n_t n w s c", n=self.vae_scale)
    middle_index = self.audio_window // 2
    latter_first_frame_audio_emb = latter_frame_audio_emb[:, :, :1, : middle_index + 1, ...]
    latter_first_frame_audio_emb = rearrange(latter_first_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")
    latter_last_frame_audio_emb = latter_frame_audio_emb[:, :, -1:, middle_index:, ...]
    latter_last_frame_audio_emb = rearrange(latter_last_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")
    latter_middle_frame_audio_emb = latter_frame_audio_emb[:, :, 1:-1, middle_index : middle_index + 1, ...]
    latter_middle_frame_audio_emb = rearrange(latter_middle_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")
    latter_frame_audio_emb_s = torch.concat(
        [latter_first_frame_audio_emb, latter_middle_frame_audio_emb, latter_last_frame_audio_emb], dim=2
    )
    first_frame_audio_emb_s = first_frame_audio_emb_s.contiguous()
    audio_embedding = self.audio_proj(first_frame_audio_emb_s, latter_frame_audio_emb_s)
    human_num = len(audio_embedding)
    audio_embedding = torch.concat(audio_embedding.split(1), dim=2).to(x.dtype)

    # convert ref_target_masks to token_ref_target_masks
    if ref_target_masks is not None:
        ref_target_masks = ref_target_masks.unsqueeze(0).to(torch.float32)
        token_ref_target_masks = nn.functional.interpolate(ref_target_masks, size=(N_h, N_w), mode="nearest")
        token_ref_target_masks = token_ref_target_masks.squeeze(0)
        token_ref_target_masks = token_ref_target_masks > 0
        token_ref_target_masks = token_ref_target_masks.view(token_ref_target_masks.shape[0], -1)
        token_ref_target_masks = token_ref_target_masks.to(x.dtype)

    # Context Parallel
    x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]

    if audio_mask is not None:
        audio_mask = torch.chunk(audio_mask, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=freqs,
        context=context,
        context_lens=context_lens,
        audio_embedding=audio_embedding,
        ref_target_masks=token_ref_target_masks,
        human_num=human_num,
        enable_sp=True,
        audio_mask=audio_mask,
    )

    for block in self.blocks:
        x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # Context Parallel
    x = get_sp_group().all_gather(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)

    return torch.stack(x)


def usp_attn_forward_avatar(
    self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16, ref_target_masks=None, human_num=None
):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    q = rope_apply(q, grid_sizes, freqs)
    k = rope_apply(k, grid_sizes, freqs)

    from yunchang.kernels import AttnType

    attn_type = AttnType.FA

    x = xFuserLongContextAttention(attn_type=attn_type, use_pack_qkv=True)(
        None, query=half(q), key=half(k), value=half(v), window_size=self.window_size
    )

    # output
    x = x.flatten(2)
    x = self.o(x)
    x_ref_attn_map = None
    if human_num != 1:
        with torch.no_grad():
            x_ref_attn_map = get_attn_map_with_target(
                q.type_as(x), k.type_as(x), grid_sizes[0], ref_target_masks=ref_target_masks, enable_sp=True
            )

    return x, x_ref_attn_map


def optimized_transform_before(x, N_t):
    x = get_sp_group().all_gather(x.to(torch.bfloat16).contiguous(), dim=1)
    # x = rearrange(x, "B (N_t S) C -> (B N_t) S C", N_t=N_t)
    S = x.shape[1] // N_t
    x = x.reshape(x.shape[0] * N_t, S, x.shape[2])
    x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    return x


def optimized_gather_after(x: torch.Tensor, N_t: int) -> torch.Tensor:
    x = get_sp_group().all_gather(x.to(torch.bfloat16).contiguous(), dim=1)
    # x = rearrange(x, "(B N_t) S C -> B (N_t S) C", N_t=N_t)
    B = x.shape[0] // N_t
    x = x.reshape(B, N_t * x.shape[1], x.shape[2])
    x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]

    return x


def usp_crossattn_multi_forward_avatar(
    self,
    x: torch.Tensor,
    encoder_hidden_states: torch.Tensor,  # 1, 21, 64, C
    shape=None,
    x_ref_attn_map=None,
    human_num=None,
    enable_sp=True,
) -> torch.Tensor:
    N_t = shape[0]
    x = optimized_transform_before(x, N_t)
    x = x if x.is_contiguous() else x.contiguous()

    sp_size = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()
    if x_ref_attn_map is not None:
        x_ref_attn_map = get_sp_group().all_gather(
            x_ref_attn_map.to(torch.bfloat16).contiguous(), dim=1
        )  # -> [H, L_global]
        S_per_frame = x_ref_attn_map.shape[1] // N_t
        x_ref_attn_map = x_ref_attn_map.view(x_ref_attn_map.shape[0], N_t, S_per_frame)
        x_ref_attn_map = torch.chunk(x_ref_attn_map, sp_size, dim=2)[sp_rank]  # -> [H, N_t, S_per_frame/sp_size]
        x_ref_attn_map = x_ref_attn_map.reshape(x_ref_attn_map.shape[0], -1).type_as(x)

    x = self.origin_forward(
        x, encoder_hidden_states, shape=shape, x_ref_attn_map=x_ref_attn_map, enable_sp=enable_sp, human_num=human_num
    )
    x = optimized_gather_after(x, N_t)
    return x
