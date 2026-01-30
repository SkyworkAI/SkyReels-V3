import torch
import torch.amp as amp
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
from xfuser.core.long_ctx_attention import xFuserLongContextAttention
from yunchang.kernels import AttnType

from ..modules.transformer import sinusoidal_embedding_1d


def pad_freqs(original_tensor, target_len):
    """Pad frequency tensor to target length for sequence parallel."""
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size, s1, s2, dtype=original_tensor.dtype, device=original_tensor.device
    )
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


@amp.autocast("cuda", enabled=False)
def rope_apply(
    x,
    grid_sizes,
    freqs,
    context_window_size=0,
    num_token_list=[],
    num_frame_list=[],
    grid_size_list=[],
):
    """
    Apply rotary position embedding with sequence parallel support.

    Args:
        x: Input tensor [B, L, N, C] where L is the SP-sliced sequence length
        grid_sizes: Grid dimensions [3] containing (F, H, W)
        freqs: Rope frequencies [M, C // 2]
        context_window_size: Window size for context support
        num_token_list: List of token counts for context videos
        num_frame_list: List of frame counts for context videos
        grid_size_list: List of grid sizes for context videos
    """
    n, c = x.size(2), x.size(3) // 2
    bs = x.size(0)
    s = x.size(1)  # SP-sliced sequence length

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    f, h, w = grid_sizes.tolist()

    # SP info
    sp_size = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()

    if context_window_size == 0 and len(num_frame_list) > 0:
        # Context support mode
        num_frame = f - sum(num_frame_list)

        latent_seq_len = num_frame * h * w
        freqs_i = torch.cat(
            [
                freqs[0][:num_frame]
                .view(num_frame, 1, 1, -1)
                .expand(num_frame, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(num_frame, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(num_frame, h, w, -1),
            ],
            dim=-1,
        ).reshape(latent_seq_len, 1, -1)

        freqs_context_list = []
        for ii, nf in enumerate(num_frame_list):
            start = 1024 if ii == 0 else 1024 + sum(num_frame_list[:ii])
            freqs_temp = torch.cat(
                [
                    freqs[0][start : start + nf]
                    .view(nf, 1, 1, -1)
                    .expand(nf, h, w, -1),
                    freqs[1][:h].view(1, h, 1, -1).expand(nf, h, w, -1),
                    freqs[2][:w].view(1, 1, w, -1).expand(nf, h, w, -1),
                ],
                dim=-1,
            ).reshape(num_token_list[ii], 1, -1)
            freqs_context_list.append(freqs_temp)

        freqs_context = torch.cat(freqs_context_list, dim=0)
        freqs_i = torch.cat([freqs_i, freqs_context], dim=0)

    elif context_window_size != 0:
        # Context window mode
        num_latent_frame = f - sum(num_frame_list)
        latent_seq_len = num_latent_frame * h * w

        freqs_list = []
        for i, nf in enumerate(num_frame_list):
            start = 0 if i == 0 else sum(num_frame_list[:i])
            _, c_h, c_w = grid_size_list[i]
            end = start + nf
            freqs_tmp = torch.cat(
                [
                    freqs[0][start:end].view(nf, 1, 1, -1).expand(nf, c_h, c_w, -1),
                    freqs[1][:c_h].view(1, c_h, 1, -1).expand(nf, c_h, c_w, -1),
                    freqs[2][:c_w].view(1, 1, c_w, -1).expand(nf, c_h, c_w, -1),
                ],
                dim=-1,
            ).reshape(nf * c_h * c_w, 1, -1)
            freqs_list.append(freqs_tmp)

        freqs_i = torch.cat(
            [
                freqs[0][sum(num_frame_list) : f]
                .view(num_latent_frame, 1, 1, -1)
                .expand(num_latent_frame, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(num_latent_frame, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(num_latent_frame, h, w, -1),
            ],
            dim=-1,
        ).reshape(latent_seq_len, 1, -1)
        freqs_list.append(freqs_i)
        freqs_i = torch.cat(freqs_list, dim=0)

    else:
        # Standard rope apply
        seq_len = f * h * w
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

    # Apply sequence parallel slicing to freqs
    freqs_i = pad_freqs(freqs_i, s * sp_size)
    freqs_i_rank = freqs_i[(sp_rank * s) : ((sp_rank + 1) * s), :, :]

    # Apply rotary embedding (matching single-card logic)
    x = torch.view_as_complex(x.to(torch.float32).reshape(bs, s, n, -1, 2))
    x = torch.view_as_real(x * freqs_i_rank.to(x.device)).flatten(3)

    return x


def usp_attn_forward(
    self,
    x,
    grid_sizes,
    freqs,
    block_mask=None,
    context_window_size=0,
    num_token_list=[],
    num_frame_list=[],
    grid_size_list=[],
):
    """
    USP (Unified Sequence Parallel) self-attention forward.

    Args:
        x: Input tensor [B, L, C]
        grid_sizes: Grid dimensions [3] containing (F, H, W)
        freqs: Rope frequencies
        block_mask: Optional attention mask
        context_window_size: Window size for context support
        num_token_list: List of token counts for context videos
        num_frame_list: List of frame counts for context videos
        grid_size_list: List of grid sizes for context videos
    """
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(torch.bfloat16)

    # query, key, value
    x = x.to(torch.bfloat16)
    q = self.norm_q(self.q(x)).view(b, s, n, d)
    k = self.norm_k(self.k(x)).view(b, s, n, d)
    v = self.v(x).view(b, s, n, d)

    # apply rope
    q = rope_apply(
        q,
        grid_sizes,
        freqs,
        context_window_size,
        num_token_list,
        num_frame_list,
        grid_size_list,
    )
    k = rope_apply(
        k,
        grid_sizes,
        freqs,
        context_window_size,
        num_token_list,
        num_frame_list,
        grid_size_list,
    )

    # attention with xFuser
    # attn_type = AttnType.SAGE_AUTO
    attn_type = AttnType.FA
    x = xFuserLongContextAttention(attn_type=attn_type)(
        None, query=half(q), key=half(k), value=half(v), window_size=self.window_size
    )

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x


def usp_dit_forward(
    self,
    x,
    t,
    context,
    clip_fea=None,
    y=None,
    block_mask=None,
    context_window_size=0,
    block_offload: bool = False,
):
    """
    USP (Unified Sequence Parallel) DiT forward.

    Args:
        x: Input video tensor [C, T, H, W] or list of tensors
        t: Diffusion timesteps [B]
        context: Text embeddings [L, C]
        clip_fea: CLIP image features (optional, for i2v)
        y: Conditional video inputs (optional, for i2v)
        block_mask: Optional attention mask
        context_window_size: Window size for context support
    """
    if self.model_type == "i2v":
        assert clip_fea is not None and y is not None

    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    # Handle both tensor and list inputs
    if isinstance(x, torch.Tensor):
        if y is not None:
            x = torch.cat([x, y], dim=1)

        # embeddings
        x = self.patch_embedding(x)
        grid_sizes = torch.tensor(x.shape[2:], dtype=torch.long)
        x = x.flatten(2).transpose(1, 2)

        grid_size_list = []
        num_frame_list = []
        num_token_list = []
    else:
        # list of tensors path
        x = [self.patch_embedding(item) for item in x]
        grid_size_list = [
            torch.tensor(item.shape[2:], dtype=torch.long) for item in x[1:]
        ]
        num_frame_list = [item.shape[2] for item in x[1:]]

        grid_sizes = torch.tensor(x[0].shape[2:], dtype=torch.long)
        grid_sizes[0] = grid_sizes[0] + sum(num_frame_list)

        x = [item.flatten(2).transpose(1, 2) for item in x]
        num_token_list = [item.shape[1] for item in x[1:]]

        x = torch.cat(x, dim=1)

    # time embeddings
    with amp.autocast("cuda", dtype=torch.float32):
        b, f = t.shape

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(
                self.patch_embedding.weight.dtype
            )
        )
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        e = e.view(b, f, -1)
        e0 = e0.view(b, f, 6, self.dim)

        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context embedding
    context = self.text_embedding(context)
    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1)

    # sequence parallel split
    sp_size = get_sequence_parallel_world_size()
    sp_rank = get_sequence_parallel_rank()
    e0 = torch.chunk(e0, sp_size, dim=1)[sp_rank]
    x = torch.chunk(x, sp_size, dim=1)[sp_rank]

    # block forward arguments
    kwargs = dict(
        e=e0,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        grid_size_list=grid_size_list,
        num_frame_list=num_frame_list,
        num_token_list=num_token_list,
        context_window_size=context_window_size,
        block_mask=block_mask,
    )

    for block in self.blocks:
        x = block(x, **kwargs)

    # head
    e = torch.chunk(e, sp_size, dim=1)[sp_rank]
    x = self.head(x, e)

    # gather from all ranks
    x = get_sp_group().all_gather(x, dim=1)

    # remove context tokens if present
    if len(num_token_list) > 0:
        num_context_token = sum(num_token_list)
        x = x[:, :-num_context_token]
        grid_sizes[0] = grid_sizes[0] - sum(num_frame_list)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return x.float()
