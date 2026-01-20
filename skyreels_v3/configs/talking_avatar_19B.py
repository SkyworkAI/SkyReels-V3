# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from easydict import EasyDict

from .shared_config import wan_shared_cfg

# ------------------------ Wan I2V 14B ------------------------#

talking_avatar_19B = EasyDict(__name__="Config: Skyreels-V3 Talking Avatar 19B")
talking_avatar_19B.update(wan_shared_cfg)
talking_avatar_19B.sample_neg_prompt = "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

talking_avatar_19B.t5_checkpoint = "models_t5_umt5-xxl-enc-bf16.pth"
talking_avatar_19B.t5_tokenizer = "google/umt5-xxl"

# clip
talking_avatar_19B.clip_model = "clip_xlm_roberta_vit_h_14"
talking_avatar_19B.clip_dtype = torch.float16
talking_avatar_19B.clip_checkpoint = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
talking_avatar_19B.clip_tokenizer = "xlm-roberta-large"

# vae
talking_avatar_19B.vae_checkpoint = "Wan2.1_VAE.pth"
talking_avatar_19B.vae_stride = (4, 8, 8)

# transformer
talking_avatar_19B.patch_size = (1, 2, 2)
talking_avatar_19B.dim = 5120
talking_avatar_19B.ffn_dim = 13824
talking_avatar_19B.freq_dim = 256
talking_avatar_19B.num_heads = 40
talking_avatar_19B.num_layers = 40
talking_avatar_19B.window_size = (-1, -1)
talking_avatar_19B.qk_norm = True
talking_avatar_19B.cross_attn_norm = True
talking_avatar_19B.eps = 1e-6
