import av
import numpy as np
import torch
from config import ASPECT_RATIO_CONFIG


def get_prefix_and_raw_video(input_video_path: str, num_condition_frames: int):
    container = av.open(input_video_path)
    stream = container.streams.video[0]
    frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    prefix_idx = list(range(len(frames)))[-num_condition_frames:]
    prefix_video = np.stack([frames[i] for i in prefix_idx])  # 形状为 [N, H, W, 3]

    raw_video_idx = list(range(len(frames)))[:-num_condition_frames]
    raw_video = np.stack([frames[i] for i in raw_video_idx])
    return prefix_video, raw_video


def get_closest_ratio(height: float, width: float, ratios: dict):
    aspect_ratio = height / width
    closest_ratio = min(
        ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio)
    )
    return closest_ratio


def process_video(prefix_video, raw_video, ASPECT_RATIO):
    # prepare for VAE
    prefix_video = (
        torch.tensor(prefix_video).permute(3, 0, 1, 2).unsqueeze(0).float()
    )  # 1, C, T, H, W
    prefix_video = prefix_video / (255.0 / 2.0) - 1.0
    prefix_video = prefix_video  # .to(pipe.device)
    if raw_video is not None:
        raw_video = (
            torch.tensor(raw_video).permute(3, 0, 1, 2).unsqueeze(0).float()
        )  # 1, C, T, H, W
        # raw_video = raw_video / (255.0 / 2.0) - 1.0
    # resize
    h, w = prefix_video.shape[-2:]
    height, width = ASPECT_RATIO[get_closest_ratio(h, w, ASPECT_RATIO)]
    height = height // 8 // 2 * 2 * 8
    width = width // 8 // 2 * 2 * 8
    prefix_video = torch.nn.functional.interpolate(
        prefix_video, size=(prefix_video.shape[2], height, width)
    )
    if raw_video is not None:
        raw_video = torch.nn.functional.interpolate(
            raw_video, size=(raw_video.shape[2], height, width)
        )
        raw_video = raw_video.squeeze(0).permute(1, 2, 3, 0).type(torch.uint8)
    return prefix_video, raw_video, height, width


def get_video_info(input_video_path: str, num_condition_frames: int, resolution: str):
    prefix_video, raw_video = get_prefix_and_raw_video(
        input_video_path, num_condition_frames
    )
    assert resolution in ASPECT_RATIO_CONFIG, f"Resolution {resolution} not supported"
    aspect_ratio = ASPECT_RATIO_CONFIG[resolution]
    prefix_video, raw_video, height, width = process_video(
        prefix_video, raw_video, aspect_ratio
    )
    return prefix_video, raw_video, height, width
