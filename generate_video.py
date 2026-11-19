import argparse
import logging
import os
import random
import time
from urllib.parse import urlsplit

# 配置日志格式和级别，实现实时终端打印
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - skyreels_v3 - %(levelname)s - [%(filename)s:%(lineno)d - %(funcName)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
    handlers=[logging.StreamHandler()],  # 显式指定输出到终端
)
import subprocess

import imageio
import torch
# torch.cuda.set_per_process_memory_fraction(0.75)
import torch.distributed as dist
import wget
from diffusers.utils import load_image

from skyreels_v3.configs import WAN_CONFIGS
from skyreels_v3.modules import download_model
from skyreels_v3.pipelines import (
    ReferenceToVideoPipeline,
    ShotSwitchingExtensionPipeline,
    SingleShotExtensionPipeline,
    TalkingAvatarPipeline,
)
from skyreels_v3.utils.avatar_preprocess import preprocess_audio


def maybe_download(path_or_url: str, save_dir: str) -> str:
    """
    If `path_or_url` is already a local path, return it.
    Otherwise, download it into `save_dir` and return the downloaded local path.
    """
    if os.path.exists(path_or_url):
        return path_or_url

    url = path_or_url
    parsed_url = urlsplit(url)
    filename = os.path.basename(parsed_url.path)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc or filename in {"", ".", ".."}:
        raise ValueError(f"Expected an existing file or an HTTP(S) URL with a filename: {path_or_url!r}")
    local_path = os.path.join(save_dir, filename)
    logging.info(f"downloading input: {local_path}")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path):
        logging.info(f"input already exists: {local_path}")
        return local_path

    wget.download(url, local_path)
    assert os.path.exists(local_path), f"Failed to download input: {url}"
    logging.info(f"finished downloading input: {local_path}")
    return local_path


def prepare_and_broadcast_inputs(args, local_rank: int):
    """
    Prepare (download) inputs on rank0, and broadcast resolved local paths to all ranks.
    This keeps multi-process inference consistent (every process sees the same args.input_*).
    """
    is_dist = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    is_rank0 = (dist.get_rank() == 0) if is_dist else (local_rank == 0)

    obj_list = [None]
    if is_rank0:
        updates = {
            "input_video": args.input_video,
            "input_audio": args.input_audio,
            "input_image": args.input_image,
            "ref_imgs": args.ref_imgs,
        }

        if args.task_type in ["single_shot_extension", "shot_switching_extension"]:
            updates["input_video"] = maybe_download(args.input_video, "input_video")

        if args.task_type == "talking_avatar":
            updates["input_audio"] = maybe_download(args.input_audio, "input_audio")
            updates["input_image"] = maybe_download(args.input_image, "input_image")

        if args.task_type == "reference_to_video":
            # Normalize to list[str] and resolve URLs to local paths on rank0.
            ref_imgs = args.ref_imgs
            if isinstance(ref_imgs, str):
                ref_imgs = [p.strip() for p in ref_imgs.split(",") if p.strip()]
            assert isinstance(ref_imgs, list) and len(ref_imgs) > 0, "ref_imgs must be a list of images"
            updates["ref_imgs"] = [maybe_download(p, "ref_imgs") for p in ref_imgs]

        obj_list[0] = updates
        print("prepare input data done")

    if is_dist:
        dist.broadcast_object_list(obj_list, src=0)
        dist.barrier()

    updates = obj_list[0]
    if updates:
        args.input_video = updates.get("input_video", args.input_video)
        args.input_audio = updates.get("input_audio", args.input_audio)
        args.input_image = updates.get("input_image", args.input_image)
        args.ref_imgs = updates.get("ref_imgs", args.ref_imgs)

    # For reference_to_video, load images on every rank after we agree on local paths.
    if args.task_type == "reference_to_video":
        ref_imgs = args.ref_imgs
        if isinstance(ref_imgs, str):
            ref_imgs = [p.strip() for p in ref_imgs.split(",") if p.strip()]
        if isinstance(ref_imgs, list) and (len(ref_imgs) == 0 or isinstance(ref_imgs[0], str)):
            ref_imgs = [load_image(p) for p in ref_imgs]
        args.ref_imgs = ref_imgs
        assert isinstance(args.ref_imgs, list) and len(args.ref_imgs) > 0, "ref_imgs must be a list of images"

    return args


MODEL_ID_CONFIG = {
    "single_shot_extension": "Skywork/SkyReels-V3-Video-Extension",
    "shot_switching_extension": "Skywork/SkyReels-V3-Video-Extension",
    "reference_to_video": "Skywork/SkyReels-V3-Reference2Video",
    "talking_avatar": "Skywork/SkyReels-V3-TalkingAvatar",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SkyReels V3: Multimodal Video Generation Model")

    # ==================== Task Selection ====================
    parser.add_argument(
        "--task_type",
        type=str,
        choices=[
            "single_shot_extension",  # Single-shot video extension (5s to 30s)
            "shot_switching_extension",  # Shot switching extension with cinematic transitions (Cut-In, Cut-Out, etc.), limited to 5s
            "reference_to_video",  # Generate video from 1-4 reference images with text prompt
            "talking_avatar",  # Generate talking avatar from portrait image and audio (up to 200s)
        ],
        help="Type of video generation task to perform.",
    )

    # ==================== Model Configuration ====================
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Model path or HuggingFace model ID. If not specified, will auto-select based on task_type. "
        "Supports: Skywork/SkyReels-V3-Reference2Video, Skywork/SkyReels-V3-Video-Extension, Skywork/SkyReels-V3-TalkingAvatar",
    )

    # ==================== Generation Parameters ====================
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Output video duration in seconds. "
        "For single_shot_extension: 5-30s; for shot_switching_extension: max 5s; for reference_to_video: recommended 5s.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A man is making his way forward slowly, leaning on a white cane to prop himself up.",
        help="Text prompt describing the desired video content. For shot_switching_extension, use prefixes like [ZOOM_IN_CUT], [ZOOM_OUT_CUT], etc.",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="720P",
        choices=["480P", "540P", "720P"],
        help="Output video resolution. Lower resolution (540P/480P) recommended for low VRAM GPUs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation. Required when using --use_usp mode.",
    )

    # ==================== Performance & Memory Options ====================
    parser.add_argument(
        "--use_usp",
        action="store_true",
        help="Enable multi-GPU parallel inference using xDiT USP (Unified Sequence Parallelism). "
        "Use with torchrun --nproc_per_node=N. Cannot be used with --low_vram.",
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Enable model offloading to reduce GPU memory usage.",
    )
    parser.add_argument(
        "--low_vram",
        action="store_true",
        help="Enable low VRAM mode with FP8 weight-only quantization and block offload. "
        "Recommended for GPUs with <24GB VRAM. Cannot be used with --use_usp.",
    )

    # ==================== Video Extension Parameters ====================
    parser.add_argument(
        "--input_video",
        type=str,
        default="https://skyreels-api.oss-accelerate.aliyuncs.com/examples/video_extension/test.mp4",
        help="[single_shot_extension/shot_switching_extension] Input video path or URL to extend.",
    )

    # ==================== Reference to Video Parameters ====================
    parser.add_argument(
        "--ref_imgs",
        type=str,
        default="https://skyreels-api.oss-accelerate.aliyuncs.com/examples/subject_reference/0_0.png",
        help="[reference_to_video] Reference images (1-4) for video generation. "
        "Supports character portraits, objects, and backgrounds. "
        "Multiple images should be comma-separated (e.g., 'img1.png,img2.png').",
    )

    # ==================== Talking Avatar Parameters ====================
    parser.add_argument(
        "--input_image",
        type=str,
        default="https://skyreels-api.oss-accelerate.aliyuncs.com/examples/talking_avatar_video/single1.png",
        help="[talking_avatar] Portrait image path or URL for avatar generation. "
        "Supports jpg/jpeg, png, gif, bmp formats. Works with real people, anime, animals, and stylized characters.",
    )
    parser.add_argument(
        "--input_audio",
        type=str,
        default="https://skyreels-api.oss-accelerate.aliyuncs.com/examples/talking_avatar_video/single_actor/huahai_5s.mp3",
        help="[talking_avatar] Driving audio path or URL. Supports mp3, wav formats. "
        "Audio duration must be <= 200 seconds. Supports multiple languages.",
    )

    args = parser.parse_args()

    if args.model_id is None:
        args.model_id = MODEL_ID_CONFIG[args.task_type]
    # init multi gpu environment
    local_rank = 0
    if args.use_usp:
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        dist.init_process_group("nccl")
        local_rank = dist.get_rank()
        torch.cuda.set_device(dist.get_rank())

        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )
    device = f"cuda:{local_rank}"
    assert not(args.use_usp and args.low_vram), "usp mode and low_vram mode cannot be used together"

    # In multi-process inference, only rank0 downloads the model; other ranks receive the resolved path via broadcast.
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        obj_list = [None]
        if dist.get_rank() == 0:
            obj_list[0] = download_model(args.model_id)
        dist.broadcast_object_list(obj_list, src=0)
        args.model_id = obj_list[0]
        dist.barrier()
    else:
        args.model_id = download_model(args.model_id)

    print(f"args.model_id: {args.model_id}")

    assert (args.use_usp and args.seed is not None) or (not args.use_usp), "usp mode need seed"
    if args.seed is None:
        random.seed(time.time())
        args.seed = int(random.randrange(4294967294))

    logging.info(f"input params: {args}")

    args = prepare_and_broadcast_inputs(args, local_rank)

    video_out = None

    # init pipeline
    if args.task_type == "single_shot_extension":
        pipe = SingleShotExtensionPipeline(model_path=args.model_id, use_usp=args.use_usp, offload=args.offload, low_vram=args.low_vram)
        video_out = pipe.extend_video(args.input_video, args.prompt, args.duration, args.seed, resolution=args.resolution)
    elif args.task_type == "shot_switching_extension":
        pipe = ShotSwitchingExtensionPipeline(model_path=args.model_id, use_usp=args.use_usp, offload=args.offload, low_vram=args.low_vram)
        video_out = pipe.extend_video(args.input_video, args.prompt, args.duration, args.seed, resolution=args.resolution)
    elif args.task_type == "reference_to_video":
        pipe = ReferenceToVideoPipeline(model_path=args.model_id, use_usp=args.use_usp, offload=args.offload, low_vram=args.low_vram)
        video_out = pipe.generate_video(args.ref_imgs, args.prompt, args.duration, args.seed, resolution=args.resolution)
    elif args.task_type == "talking_avatar":
        config = WAN_CONFIGS["talking-avatar-19B"]
        pipe = TalkingAvatarPipeline(
            config=config,
            model_path=args.model_id,
            device_id=local_rank,
            rank=local_rank,
            use_usp=args.use_usp,
            offload=args.offload,
            low_vram=args.low_vram,
        )
        input_data = {
            "prompt": args.prompt,
            "cond_image": args.input_image,
            "cond_audio": {"person1": args.input_audio},
        }
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            # Only rank0 does the heavy audio preprocess + file writes, then broadcasts the result.
            obj_list = [None]
            if dist.get_rank() == 0:
                input_data, _ = preprocess_audio(args.model_id, input_data, "processed_audio")
                obj_list[0] = input_data
            dist.broadcast_object_list(obj_list, src=0)
            input_data = obj_list[0]
            dist.barrier()
        else:
            input_data, _ = preprocess_audio(args.model_id, input_data, "processed_audio")
        kwargs = {
            "input_data": input_data,
            "size_buckget": args.resolution,
            "motion_frame": 5,
            "frame_num": 81,
            "drop_frame": 12,
            "shift": 11,
            "text_guide_scale": 1.0,
            "audio_guide_scale": 1.0,
            "seed": args.seed,
            "sampling_steps": 4,
            "max_frames_num": 5000,
        }
        print(f"generate video kwargs: {kwargs}")
        video_out = pipe.generate(**kwargs)
    else:
        raise ValueError(f"Invalid task type: {args.task_type}")

    save_dir = os.path.join("result", args.task_type)
    os.makedirs(save_dir, exist_ok=True)

    if local_rank == 0:
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        video_out_file = f"{args.seed}_{current_time}.mp4"
        output_path = os.path.join(save_dir, video_out_file)
        fps = 25 if args.task_type == "talking_avatar" else 24
        imageio.mimwrite(
            output_path,
            video_out,
            fps=fps,
            quality=8,
            output_params=["-loglevel", "error"],
        )
        if args.task_type == "talking_avatar":
            video_with_audio_path = os.path.join(save_dir, video_out_file.replace(".mp4", "_with_audio.mp4"))
            audio_path = kwargs["input_data"]["video_audio"]
            video_in = os.path.abspath(output_path)
            audio_in = os.path.abspath(audio_path)
            video_out_with_audio = os.path.abspath(video_with_audio_path)
            print(f"video_in: {video_in}, audio_in: {audio_in}, video_out_with_audio: {video_out_with_audio}")
            # fmt: off
            cmd = [
                'ffmpeg',
                '-y',
                '-i', f'"{video_in}"',
                '-i', f'"{audio_in}"',
                '-map', '0:v',
                '-map', '1:a',
                '-c:v', 'copy',
                '-shortest',
                f'"{video_out_with_audio}"'
            ]
            # fmt: on

            try:
                subprocess.run(
                    " ".join(cmd),
                    shell=True,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                print(f"Video with audio generated successfully: {video_with_audio_path}")
                os.remove(video_in) # remove the original video
            except subprocess.CalledProcessError as e:
                print(f"ffmpeg failed (exit={e.returncode}). Output:\n{e.stdout}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
