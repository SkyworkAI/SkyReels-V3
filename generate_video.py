import argparse
import logging
import os
import random
import time

import imageio
import torch
import wget

# 配置日志格式和级别，实现实时终端打印
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - skyreels_v3 - %(levelname)s - [%(filename)s:%(lineno)d - %(funcName)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
    handlers=[logging.StreamHandler()],  # 显式指定输出到终端
)
from diffusers.utils import load_image

from skyreels_v3.modules import download_model
from skyreels_v3.pipelines import (
    ReferenceToVideoPipeline,
    ShotSwitchingExtensionPipeline,
    SingleShotExtensionPipeline,
)


MODEL_ID_CONFIG = {
    "single_shot_extension": "Skywork/SkyReels-V3-Video-Extension",
    "shot_switching_extension": "Skywork/SkyReels-V3-Video-Extension",
    "reference_to_video": "Skywork/SkyReels-V3-Reference2Video",
    "talking_avatar": "Skywork/SkyReels-V3-TalkingAvatar",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_type",
        type=str,
        choices=[
            "single_shot_extension",
            "shot_switching_extension",
            "reference_to_video",
            "talking_avatar",
        ],
    )
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument(
        "--ref_imgs",
        type=str,
        default="https://skyreels-api.oss-accelerate.aliyuncs.com/examples/subject_reference/0_0.png",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A man is making his way forward slowly, leaning on a white cane to prop himself up.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_usp", action="store_true")
    parser.add_argument("--offload", action="store_true")
    parser.add_argument(
        "--input_video",
        type=str,
        default="https://skyreels-api.oss-accelerate.aliyuncs.com/examples/video_extension/test.mp4",
    )

    args = parser.parse_args()

    if args.model_id is None:
        args.model_id = download_model(MODEL_ID_CONFIG[args.task_type])
    else:
        args.model_id = download_model(args.model_id)
    assert (args.use_usp and args.seed is not None) or (
        not args.use_usp
    ), "usp mode need seed"
    if args.seed <= 0 and not args.use_usp:
        random.seed(time.time())
        args.seed = int(random.randrange(4294967294))

    if args.task_type in ["single_shot_extension", "shot_switching_extension"]:
        # check if args.input_video is a local video file, otherwise download it
        if not os.path.exists(args.input_video):
            video_url = args.input_video
            video_name = video_url.split("/")[-1]
            video_path = os.path.join("input_video", video_name)
            logging.info(f"downloading input video: {video_path}")
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            if os.path.exists(video_path):
                logging.info(f"input video already exists: {video_path}")
                args.input_video = video_path
            else:
                wget.download(video_url, video_path)
                assert os.path.exists(
                    args.input_video
                ), f"Failed to download input video: {args.input_video}"
                logging.info(f"finished downloading input video: {args.input_video}")
    elif args.task_type == "reference_to_video":
        ref_imgs = args.ref_imgs.split(",")
        args.ref_imgs = [load_image(img.strip()) for img in ref_imgs]
        assert len(args.ref_imgs) > 0, "ref_imgs must be a list of images"
    else:
        raise ValueError(f"Invalid task type: {args.task_type}")
    
    args.model_id = MODEL_ID_CONFIG[args.task_type]

    logging.info(f"input params: {args}")

    # init multi gpu environment
    local_rank = 0
    if args.use_usp:
        import torch.distributed as dist
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        dist.init_process_group("nccl")
        local_rank = dist.get_rank()
        torch.cuda.set_device(dist.get_rank())

        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size()
        )

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )
    device = f"cuda:{local_rank}"
    video_out = None

    # init pipeline
    if args.task_type == "single_shot_extension":
        pipe = SingleShotExtensionPipeline(
            model_path=args.model_id, use_usp=args.use_usp, offload=args.offload
        )
        video_out = pipe.extend_video(
            args.input_video, args.prompt, args.duration, args.seed
        )
    elif args.task_type == "shot_switching_extension":
        pipe = ShotSwitchingExtensionPipeline(
            model_path=args.model_id, use_usp=args.use_usp, offload=args.offload
        )
        video_out = pipe.extend_video(
            args.input_video, args.prompt, args.duration, args.seed
        )
    elif args.task_type == "reference_to_video":
        pipe = ReferenceToVideoPipeline(
            model_path=args.model_id, use_usp=args.use_usp, offload=args.offload
        )
        video_out = pipe.generate_video(
            args.ref_imgs, args.prompt, args.duration, args.seed
        )
    else:
        raise ValueError(f"Invalid task type: {args.task_type}")

    save_dir = os.path.join("result", args.task_type)
    os.makedirs(save_dir, exist_ok=True)

    if local_rank == 0:
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        video_out_file = (
            f"{args.prompt[:100].replace('/','')}_{args.seed}_{current_time}.mp4"
        )
        output_path = os.path.join(save_dir, video_out_file)
        imageio.mimwrite(
            output_path,
            video_out,
            fps=24,
            quality=8,
            output_params=["-loglevel", "error"],
        )
