import gc
import html
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ftfy
import numpy as np
import regex as re
import torch
from diffusers import UniPCMultistepScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.models import AutoencoderKLWan, WanTransformer3DModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from PIL import Image, ImageOps
from torchvision.transforms import functional as F
from transformers import AutoTokenizer, UMT5EncoderModel

from skyreels_v3.modules.reference_to_video.transformer import SkyReelsA2WanI2v3DModel

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

MAX_ALLOWED_REF_IMG_LENGTH = 4


def basic_clean(text):
    """Basic text cleaning: fix encoding and decode HTML entities."""
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    """Remove redundant whitespace."""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    """Complete cleaning pipeline for prompts."""
    text = whitespace_clean(basic_clean(text))
    return text


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
):
    """Extract latents from encoder output."""
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class WanSkyReelsA2WanT2VPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    r"""
    Pipeline for image-to-video generation using Wan model.

    Args:
        tokenizer ([`T5Tokenizer`]):
            Tokenizer from T5, specifically the google/umt5-xxl variant.
        text_encoder ([`T5EncoderModel`]):
            T5 text encoder, specifically the google/umt5-xxl variant.
        transformer ([`WanTransformer3DModel`]):
            Conditional Transformer to denoise the input latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded video latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        transformer: WanTransformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_scale_factor_temporal = (
            2 ** sum(self.vae.temperal_downsample) if self.vae else 4
        )
        self.vae_scale_factor_spatial = (
            2 ** len(self.vae.temperal_downsample) if self.vae else 8
        )
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Get T5 prompt embeddings."""
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype
        logging.info(
            f"reference_to_video_pipeline _get_t5_prompt_embeds device: {device}, dtype: {dtype} {self.text_encoder.device}"
        )

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), mask.to(device)
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                for u in prompt_embeds
            ],
            dim=0,
        )

        # Duplicate text embeddings for each generation per prompt
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_videos_per_prompt, seq_len, -1
        )

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Encode prompts into hidden states."""
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = (
                batch_size * [negative_prompt]
                if isinstance(negative_prompt, str)
                else negative_prompt
            )

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        """Check the validity of input arguments."""
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 16 but are {height} and {width}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs
            for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
            not isinstance(prompt, str) and not isinstance(prompt, list)
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str)
            and not isinstance(negative_prompt, list)
        ):
            raise ValueError(
                f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}"
            )

    def prepare_latents(
        self,
        image_vae,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare initial latents and reference image latents."""
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            latent_height,
            latent_width,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device=device, dtype=dtype)

        ref_vae_latents = []
        for ref_image in image_vae:
            ref_image = F.to_tensor(ref_image).sub_(0.5).div_(0.5).to(device)
            img_vae_latent = self.vae.encode(
                ref_image.unsqueeze(1).unsqueeze(0)
            )  # 1*3*1*H*W
            img_vae_latent = retrieve_latents(img_vae_latent, generator)

            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                1, self.vae.config.z_dim, 1, 1, 1
            ).to(latents.device, latents.dtype)

            img_vae_latent = (img_vae_latent - latents_mean) * latents_std

            ref_vae_latents.append(img_vae_latent)

        assert (
            len(ref_vae_latents) <= MAX_ALLOWED_REF_IMG_LENGTH
        ), f"ref_vae_latents length is {len(ref_vae_latents)}, but MAX_ALLOWED_REF_IMG_LENGTH is {MAX_ALLOWED_REF_IMG_LENGTH}"

        # Pad with empty latents to match maximum allowed reference image length
        while len(ref_vae_latents) < MAX_ALLOWED_REF_IMG_LENGTH:
            empty_latent = torch.zeros(
                1, num_channels_latents, 1, latent_height, latent_width
            ).to(device=device, dtype=dtype)
            ref_vae_latents.append(empty_latent)

        ref_vae_latents = torch.cat(ref_vae_latents, dim=2)  # 1*c*T*H*W
        ref_vae_latents = ref_vae_latents.repeat(batch_size, 1, 1, 1, 1)

        return latents, ref_vae_latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def _execution_device(self):
        device_id = os.environ.get("LOCAL_RANK", 0)
        return f"cuda:{device_id}"

    @torch.no_grad()
    def __call__(
        self,
        ref_imgs,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 544,
        width: int = 960,
        num_frames: int = 97 + 8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        guidance_scale_img: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        start_step=0,
        offload=False,
    ):
        r"""
        The call function for video generation.

        Args:
            ref_imgs (`List[PIL.Image]`):
                List of reference images to guide generation.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide video generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide video generation.
            height (`int`, defaults to `544`):
                The height of the generated video.
            width (`int`, defaults to `960`):
                The width of the generated video.
            num_frames (`int`, defaults to `105`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps.
            guidance_scale (`float`, defaults to `7.5`):
                Guidance scale for text classifier-free guidance.
            guidance_scale_img (`float`, defaults to `5.0`):
                Guidance scale for image classifier-free guidance.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A random number generator to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated video.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a WanPipelineOutput instead of a plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function that is called at the end of each denoising step.
            max_sequence_length (`int`, *optional*, defaults to `512`):
                Maximum sequence length of the prompt.
        """
        if offload:
            self.text_encoder.to(self._execution_device)

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define batch size
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode prompts
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if offload:
            self.text_encoder.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            self.transformer.to(self._execution_device)

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latents
        num_channels_latents = self.vae.config.z_dim
        latents, condition = self.prepare_latents(
            ref_imgs,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )
        uncondition = torch.zeros_like(condition)

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                timestep = t.expand(latents.shape[0])

                # Forward prediction
                noise_pred = self.transformer(
                    hidden_states=torch.cat([latents, condition], dim=2).to(
                        transformer_dtype
                    ),
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, :, : latents.shape[2], :, :]

                if self.do_classifier_free_guidance:
                    # Text unconditioned prediction
                    noise_uncond_txt = self.transformer(
                        hidden_states=torch.cat([latents, condition], dim=2).to(
                            transformer_dtype
                        ),
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_uncond_txt = noise_uncond_txt[:, :, : latents.shape[2], :, :]

                    # Text + Image unconditioned prediction
                    noise_uncond_txt_img = self.transformer(
                        hidden_states=torch.cat([latents, uncondition], dim=2).to(
                            transformer_dtype
                        ),
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_uncond_txt_img = noise_uncond_txt_img[
                        :, :, : latents.shape[2], :, :
                    ]

                    # Apply dual guidance logic
                    noise_pred = (
                        noise_uncond_txt_img
                        + guidance_scale_img * (noise_uncond_txt - noise_uncond_txt_img)
                        + guidance_scale * (noise_pred - noise_uncond_txt)
                    )

                # Update latents
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None
        if offload:
            self.transformer.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

        # 7. Decode latents
        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                1, self.vae.config.z_dim, 1, 1, 1
            ).to(latents.device, latents.dtype)
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(
                video, output_type=output_type
            )
        else:
            video = latents

        # Offload models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)


def resize_ref_images(ref_imgs, size):
    # Load size.
    h, w = size[1], size[0]
    # Load images.
    ref_images = []
    for img in ref_imgs:
        img = img.convert("RGB")

        # Calculate the required size to keep aspect ratio and fill the rest with padding.
        img_ratio = img.width / img.height
        target_ratio = w / h

        if img_ratio > target_ratio:  # Image is wider than target
            new_width = w
            new_height = int(new_width / img_ratio)
        else:  # Image is taller than target
            new_height = h
            new_width = int(new_height * img_ratio)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create a new image with the target size and place the resized image in the center
        delta_w = w - img.size[0]
        delta_h = h - img.size[1]
        padding = (
            delta_w // 2,
            delta_h // 2,
            delta_w - (delta_w // 2),
            delta_h - (delta_h // 2),
        )
        new_img = ImageOps.expand(img, padding, fill=(255, 255, 255))
        ref_images.append(new_img)

    return ref_images


class ReferenceToVideoPipeline:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        weight_dtype=torch.bfloat16,
        use_usp=False,
        offload=False,
    ):
        """
        Initialize the reference to video pipeline class

        Args:
            model_path (str): Path to the model
            device (str): Device to run on, defaults to 'cuda'
            weight_dtype: Weight data type, defaults to torch.bfloat16
            use_usp: Whether to use USP, defaults to False
            offload: Whether to offload the model to CPU, defaults to False
        """
        load_device = "cpu" if offload else device
        self.transformer = SkyReelsA2WanI2v3DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=torch.bfloat16
        ).to(load_device)
        self.vae = AutoencoderKLWan.from_pretrained(
            model_path, subfolder="vae", torch_dtype=torch.float32
        ).to(load_device)

        self.pipeline = WanSkyReelsA2WanT2VPipeline.from_pretrained(
            model_path,
            transformer=self.transformer,
            vae=self.vae,
            torch_dtype=weight_dtype,
        ).to(load_device)

        self.pipeline.scheduler = UniPCMultistepScheduler(
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            num_train_timesteps=1000,
            flow_shift=5.0,
        )

        self.use_usp = use_usp
        self.offload = offload
        self.device = device

        # hardcode
        from torchao.quantization import float8_weight_only, quantize_

        quantize_(self.pipeline.transformer, float8_weight_only(), device=device)
        self.pipeline.transformer.to(load_device)

        if self.use_usp:
            from ..distributed.context_parallel_for_reference import (
                parallelize_transformer,
            )

            parallelize_transformer(self.pipeline)

        if self.offload:
            self.pipeline.vae.to(device)
            self.pipeline.transformer.to("cpu")
            self.pipeline.text_encoder.to("cpu")
        else:
            self.pipeline.to(device)
        gc.collect()
        torch.cuda.empty_cache()

    def generate_video(self, ref_imgs, prompt, duration, seed, resolution="720P"):
        from ..utils.util import get_height_width_from_image

        height, width = get_height_width_from_image(ref_imgs[0], resolution)
        height, width = 720, 1280
        ref_imgs = resize_ref_images(ref_imgs, (width, height))
        num_frames = duration * 24 + 1
        logging.info(f"height: {height}, width: {width}, num_frames: {num_frames}")
        kwargs = {
            "ref_imgs": ref_imgs,
            "prompt": prompt,
            "negative_prompt": "",
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "guidance_scale": 1.0,
            "guidance_scale_img": 1.0,
            "generator": torch.Generator(device=self.device).manual_seed(seed),
            "output_type": "pt",
            "start_step": 0,
            "num_inference_steps": 8,
            "offload": self.offload,
        }
        logging.info(f"kwargs: {kwargs}")
        video_pt = self.pipeline(**kwargs).frames

        gc.collect()
        torch.cuda.empty_cache()

        batch_size = video_pt.shape[0]
        batch_video_frames = []
        for batch_idx in range(batch_size):
            pt_image = video_pt[batch_idx]
            pt_image = torch.stack([pt_image[i] for i in range(pt_image.shape[0])])
            image_np = VaeImageProcessor.pt_to_numpy(pt_image)
            image_pil = VaeImageProcessor.numpy_to_pil(image_np)
            batch_video_frames.append(image_pil)

        # Export the generated frames to a video file.
        video_generate = batch_video_frames[0]
        final_images = []
        for frame in video_generate:
            frame = Image.fromarray(np.array(frame)).convert("RGB")
            final_images.append(np.array(frame))
        return final_images
