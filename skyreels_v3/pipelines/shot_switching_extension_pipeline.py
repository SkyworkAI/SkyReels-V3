import gc
import os
from typing import List, Optional, Union

import numpy as np
import torch
from diffusers.video_processor import VideoProcessor
from tqdm import tqdm

from ..config import SHOT_NUM_CONDITION_FRAMES_MAP
from ..modules import get_text_encoder, get_transformer, get_vae
from ..scheduler.fm_solvers_unipc import FlowUniPCMultistepScheduler
from ..utils.util import get_video_info


class ShotSwitchingExtensionPipeline:
    """
    A pipeline for shot switching video extension tasks.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        weight_dtype=torch.bfloat16,
        use_usp=False,
        offload=False,
    ):
        """
        Initialize the diffusion forcing pipeline class

        Args:
            model_path (str): Path to the model
            device (str): Device to run on, defaults to 'cuda'
            weight_dtype: Weight data type, defaults to torch.bfloat16
        """
        load_device = "cpu" if offload else device
        self.transformer = get_transformer(model_path, subfolder="transformer", device=load_device, weight_dtype=weight_dtype)
        vae_model_path = os.path.join(model_path, "Wan2.1_VAE.pth")
        self.vae = get_vae(vae_model_path, device=device, weight_dtype=torch.float32)
        self.text_encoder = get_text_encoder(model_path, device=load_device, weight_dtype=weight_dtype)
        self.video_processor = VideoProcessor(vae_scale_factor=16)
        self.device = device
        self.offload = offload
        self.sp_size = 1

        if use_usp:
            import types

            from xfuser.core.distributed import get_sequence_parallel_world_size

            from ..distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )

            for block in self.transformer.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn
                )
                self.transformer.forward = types.MethodType(
                    usp_dit_forward, self.transformer
                )
                self.sp_size = get_sequence_parallel_world_size()

        self.scheduler = FlowUniPCMultistepScheduler()
        self.vae_stride = (4, 8, 8)
        self.patch_size = (1, 2, 2)
        self.components = {
            "vae": self.vae.vae,
            "text_encoder": self.text_encoder,
            "transformer": self.transformer,
        }
        self.offload = offload
        self.vae.to(self.device)
        if self.offload:
            self.text_encoder.to("cpu")
            self.transformer.to("cpu")
        else:
            self.text_encoder.to(self.device)
            self.transformer.to(self.device)

    def extend_video(
        self,
        raw_video: str,
        prompt: str,
        duration: int,
        seed: int,
        fps: int = 24,
        resolution: str = "720P",
    ):
        assert (
            duration in SHOT_NUM_CONDITION_FRAMES_MAP
        ), f"Duration {duration} not supported"
        num_condition_frames = SHOT_NUM_CONDITION_FRAMES_MAP[duration]
        frames_num = duration * fps + 1
        prefix_video, raw_video, height, width = get_video_info(
            raw_video, num_condition_frames, resolution
        )

        prefix_video = prefix_video.to(self.device)
        prefix_video = self.vae.encode(prefix_video)
        video_frames = self.__call__(
            prompt=prompt,
            negative_prompt="",
            width=width,
            height=height,
            num_frames=frames_num,
            num_inference_steps=8,
            guidance_scale=1.0,
            shift=8.0,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            prefix_video=prefix_video,
        )
        return video_frames

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self._guidance_scale > 1.0

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        width: int = 544,
        height: int = 960,
        num_frames: int = 97,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        shift: float = 5.0,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ):
        self._guidance_scale = guidance_scale
        if self.offload:
            self.text_encoder.to(self.device)
        # preprocess
        F = num_frames
        target_shape = (
            self.vae.vae.z_dim,
            (F - 1) // self.vae_stride[0] + 1,
            height // self.vae_stride[1],
            width // self.vae_stride[2],
        )
        context = self.text_encoder.encode(prompt).to(self.device)
        context_null = (
            self.text_encoder.encode(negative_prompt).to(self.device)
            if self.do_classifier_free_guidance
            else None
        )
        if self.offload:
            self.text_encoder.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

        latents = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=generator,
            )
        ]

        prefix_video = kwargs["prefix_video"].to(self.device)

        context_frames = prefix_video.shape[2]
        total_frames = latents[0].shape[1] + context_frames

        if self.offload:
            self.transformer.to(self.device)
        with torch.cuda.amp.autocast(dtype=self.transformer.dtype), torch.no_grad():
            self.scheduler.set_timesteps(
                num_inference_steps, device=self.device, shift=shift
            )
            timesteps = self.scheduler.timesteps

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = torch.stack(latents)
                timestep = t.repeat(latent_model_input.shape[0])
                timestep = timestep.unsqueeze(-1).repeat(1, total_frames)
                timestep[:, -context_frames:] = 0
                if guidance_scale > 1.0:
                    noise_pred_cond = self.transformer(
                        [latent_model_input, prefix_video], t=timestep, context=context
                    )[0]
                    noise_pred_uncond = self.transformer(
                        [latent_model_input, prefix_video],
                        t=timestep,
                        context=context_null,
                    )[0]
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )
                else:
                    noise_pred = self.transformer(
                        [latent_model_input, prefix_video], t=timestep, context=context
                    )[0]

                temp_x0 = self.scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=generator,
                )[0]
                latents = [temp_x0.squeeze(0)]
            if self.offload:
                self.transformer.cpu()
                torch.cuda.empty_cache()
            videos = self.vae.decode(latents[0])
            videos = (videos / 2 + 0.5).clamp(0, 1)
            videos = [video for video in videos]
            videos = [video.permute(1, 2, 3, 0) * 255 for video in videos]
            videos = [video.cpu().numpy().astype(np.uint8) for video in videos]
        if self.offload:
            gc.collect()
            torch.cuda.empty_cache()
        return videos
