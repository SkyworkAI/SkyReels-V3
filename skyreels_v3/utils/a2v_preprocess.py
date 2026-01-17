import copy
import os
import time

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
from einops import rearrange
from transformers import Wav2Vec2FeatureExtractor

from ..modules.wav2vec2 import Wav2Vec2Model


def loudness_norm(audio_array, sr=16000, lufs=-23):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio


def audio_prepare_multi_new(cond_audios, sample_rate=16000):

    human_speech_arrays = []

    try:
        for caudio in cond_audios:
            human_speech = audio_prepare_single(caudio)
            human_speech_arrays.append(human_speech)
    except:
        cond_audios = sorted(cond_audios.items(), key=lambda item: int(item[0].replace("person", "")))
        for key, caudio in cond_audios:
            human_speech = audio_prepare_single(caudio)
            human_speech_arrays.append(human_speech)

    sum_human_speechs = np.concatenate(human_speech_arrays)

    return human_speech_arrays, sum_human_speechs


def get_embedding(speech_array, wav2vec_feature_extractor, audio_encoder, sr=16000, device="cpu"):
    audio_duration = len(speech_array) / sr
    video_length = audio_duration * 25  # Assume the video fps is 25

    # wav2vec_feature_extractor
    audio_feature = np.squeeze(wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values)
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
    audio_feature = audio_feature.unsqueeze(0)

    # audio encoder
    with torch.no_grad():
        embeddings = audio_encoder(audio_feature, seq_len=int(np.ceil(video_length)), output_hidden_states=True)

    if len(embeddings) == 0:
        print("Fail to extract audio embedding")
        return None

    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")

    audio_emb = audio_emb.cpu().detach()
    return audio_emb


def audio_prepare_single(audio_path, sample_rate=16000):
    human_speech_array, sr = librosa.load(audio_path, sr=sample_rate)
    # human_speech_array: np.ndarray, shape: (duration * sr,), float32.
    # 最大内存: 12.2G, (200s)
    audio_duration = len(human_speech_array) / sr
    if audio_duration < 0.4:
        raise ValueError(f"Audio duration is too short: {audio_duration}s. Minimum allowed: 0.4s.")
    human_speech_array = loudness_norm(human_speech_array, sr)
    return human_speech_array


def preprocess_audio(model_path, input_data, audio_save_dir):

    def custom_init(device, wav2vec):
        audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec, local_files_only=True).to(device)
        audio_encoder.feature_extractor._freeze_parameters()
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec, local_files_only=True)
        return wav2vec_feature_extractor, audio_encoder

    w2v_path = os.path.join(model_path, "chinese-wav2vec2-base")
    wav2vec_feature_extractor, audio_encoder = custom_init("cpu", w2v_path)
    os.makedirs(audio_save_dir, exist_ok=True)

    return _preprocess_audio(wav2vec_feature_extractor, audio_encoder, input_data, audio_save_dir)


def _preprocess_audio(wav2vec_feature_extractor, audio_encoder, input_data, audio_save_dir):
    # if len(input_data["cond_audio"]) >= 2:
    return _preprocess_audio_multi(wav2vec_feature_extractor, audio_encoder, input_data, audio_save_dir)
    # else:
    # return _preprocess_audio_single(
    #     wav2vec_feature_extractor, audio_encoder, input_data, audio_save_dir
    # )


def _preprocess_audio_single(wav2vec_feature_extractor, audio_encoder, input_data, audio_save_dir):
    start = time.time()
    os.makedirs(audio_save_dir, exist_ok=True)
    max_frames_num = input_data.get("max_frames_num", 5000)
    _ext_info = {}

    if len(input_data["cond_audio"]) >= 2:
        raise ValueError("Single audio is not supported for multi-person audio")

    elif len(input_data["cond_audio"]) == 1:
        human_speech = audio_prepare_single(input_data["cond_audio"]["person1"])
        audio_embedding = get_embedding(human_speech, wav2vec_feature_extractor, audio_encoder)
        final_video_length = audio_embedding.shape[0] if audio_embedding.shape[0] < max_frames_num else max_frames_num

        emb_path = os.path.join(audio_save_dir, "1.pt")
        sum_audio = os.path.join(audio_save_dir, "sum.wav")
        sf.write(sum_audio, human_speech, 16000)
        torch.save(audio_embedding, emb_path)
        # 使用原始音频文件来拼接输出视频
        input_data["video_audio"] = input_data["cond_audio"]["person1"]
        input_data["cond_audio"]["person1"] = emb_path

    _ext_info["final_video_length"] = final_video_length
    print(f"preprocess audio time: {time.time()-start:0.2f}s")
    return input_data, _ext_info


def _preprocess_audio_multi(wav2vec_feature_extractor, audio_encoder, input_data, audio_save_dir):
    input_data = copy.deepcopy(input_data)
    fps = 25
    sample_rate = 16000

    start = time.time()
    os.makedirs(audio_save_dir, exist_ok=True)
    max_frames_num = input_data.get("max_frames_num", 5000)
    max_duration = max_frames_num / fps
    _ext_info = {}

    speech_list, sum_human_speechs = audio_prepare_multi_new(input_data["cond_audio"])
    audio_duration = len(sum_human_speechs) / sample_rate
    if audio_duration > max_duration:
        raise ValueError(f"Sum of audio duration is too long: {audio_duration:.2f}s. Maximum allowed: {max_duration}s")
    audio_emb_list = []
    trans_list = []
    for speech in speech_list:
        audio_embedding = get_embedding(speech, wav2vec_feature_extractor, audio_encoder)
        audio_emb_list.append(audio_embedding)
        trans_list.append(len(torch.cat(audio_emb_list, dim=0)))

    audio_emb_path_list = []
    for i, audio_embedding in enumerate(audio_emb_list):
        emb_path = os.path.join(audio_save_dir, f"{i+1}.pt")
        torch.save(audio_embedding, emb_path)
        audio_emb_path_list.append(emb_path)

    sum_audio = os.path.join(audio_save_dir, f"sum.wav")

    print("sum_human_speechs:", len(sum_human_speechs))
    print("sum_audio:", sum_audio)

    sf.write(sum_audio, sum_human_speechs, 16000)
    input_data["video_audio"] = sum_audio
    input_data["audio_embs"] = audio_emb_path_list
    input_data["trans_points"] = trans_list

    ##兼容一下单人的代码
    input_data["cond_audio"]["person1"] = audio_emb_path_list[0]

    ##兼容一下不同格式的输入
    if "bbox" in input_data:
        if type(input_data["bbox"]) == dict:
            bboxes = sorted(
                input_data["bbox"].items(),
                key=lambda item: int(item[0].replace("person", "")),
            )
            input_data["bbox"] = [box for (key, box) in bboxes]

        assert len(input_data["bbox"]) == len(input_data["cond_audio"])
    ###########

    if len(input_data["cond_audio"]) > 1:
        silent_speech = np.zeros(sum_human_speechs.shape[0])
        silent_audio_embedding = get_embedding(silent_speech, wav2vec_feature_extractor, audio_encoder)
        silent_emb_path = os.path.join(audio_save_dir, "silent.pt")
        torch.save(silent_audio_embedding, silent_emb_path)
        input_data["silent_audio_embs"] = silent_emb_path
    else:
        input_data["silent_audio_embs"] = None

    # final_video_length = (
    #     len(sum_human_speechs)
    #     if len(sum_human_speechs) < max_frames_num
    #     else max_frames_num
    # )

    _ext_info["final_video_length"] = trans_list[-1]

    print(f"preprocess audio time: {time.time()-start:0.2f}s")
    return input_data, _ext_info


CAMERA_PARAMETERS = {
    "static": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # 静止
    "push_in": [0.0, 0.0, 0.0, 0.0, 0.0, 0.6],  # 推进
    "push_out": [0.0, 0.0, 0.0, 0.0, 0.0, 1.67],  # 拉远
    "pan_left": [-0.2, 0.0, 0.0, 0.0, 0.0, 1.0],  # 向左平移
    "pan_right": [0.2, 0.0, 0.0, 0.0, 0.0, 1.0],  # 向右平移
    "crane_up": [0.0, 0.0, 0.0, -20.0, 0.0, 1.15],  # 抬升
    "crane_down": [0.0, 0.0, 0.0, 20.0, 0.0, 1.15],  # 下降
    "left_rotation": [0.0, 0.0, 0.0, 0.0, 60.0, 1.0],  # 左旋转
    "right_rotation": [0.0, 0.0, 0.0, 0.0, -60.0, 1.0],  # 右旋转
    "swing": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # 摆动, 特殊处理
}


def preprocess_camera_control_info(camera_control_info, nframe):
    new_camera_control_info = []
    is_swing = False
    for info in camera_control_info:
        if len(info) == 4:
            start_time = info[0]
            end_time = info[1]
            motion_type = info[2]  # 运镜类型
            strength = info[3]  # 运镜强度

            # 时间转换为帧索引
            start_frame_index = max(0, int(start_time * 25))  # fps=25
            if start_frame_index > nframe - 1:
                continue
            if end_time == -1:
                end_frame_index = nframe - 1  # 最后一帧
            else:
                end_frame_index = min(nframe - 1, int(end_time * 25))

            # 从CAMERA_PARAMETERS获取基础参数
            if motion_type in CAMERA_PARAMETERS:
                if motion_type == "swing":
                    is_swing = True
                base_params = CAMERA_PARAMETERS[motion_type]
                x_offset, y_offset, z_offset, d_theta, d_phi, d_r = base_params

                # 根据强度计算新的运镜参数
                new_x_offset = x_offset * strength
                new_y_offset = y_offset * strength
                new_z_offset = z_offset * strength
                new_d_theta = d_theta * strength
                new_d_phi = d_phi * strength
                new_d_r = (d_r - 1.0) * strength + 1.0
            else:
                raise ValueError(f"Unknown motion type: {motion_type}")

            # 构建8位参数格式: [start_frame_index, end_frame_index, x_offset, y_offset, z_offset, d_theta, d_phi, d_r]
            new_info = [
                start_frame_index,
                end_frame_index,
                new_x_offset,
                new_y_offset,
                new_z_offset,
                new_d_theta,
                new_d_phi,
                new_d_r,
            ]
            new_camera_control_info.append(new_info)

        elif len(info) == 6:
            start_time = info[0]
            end_time = info[1]
            motion_type1 = info[2]  # 运镜类型
            strength1 = info[3]  # 运镜强度
            motion_type2 = info[4]  # 运镜类型
            strength2 = info[5]  # 运镜强度

            # 时间转换为帧索引
            start_frame_index = max(0, int(start_time * 25))  # fps=25
            if start_frame_index > nframe - 1:
                continue
            if end_time == -1:
                end_frame_index = nframe - 1  # 最后一帧
            else:
                end_frame_index = min(nframe - 1, int(end_time * 25))

            # 从CAMERA_PARAMETERS获取基础参数
            if motion_type1 in CAMERA_PARAMETERS and motion_type2 in CAMERA_PARAMETERS:
                if motion_type1 == "swing" or motion_type2 == "swing":
                    is_swing = True
                base_params1 = CAMERA_PARAMETERS[motion_type1]
                x_offset1, y_offset1, z_offset1, d_theta1, d_phi1, d_r1 = base_params1

                base_params2 = CAMERA_PARAMETERS[motion_type2]
                x_offset2, y_offset2, z_offset2, d_theta2, d_phi2, d_r2 = base_params2

                # 根据强度计算新的运镜参数
                new_x_offset = x_offset1 * strength1 + x_offset2 * strength2
                new_y_offset = y_offset1 * strength1 + y_offset2 * strength2
                new_z_offset = z_offset1 * strength1 + z_offset2 * strength2
                new_d_theta = d_theta1 * strength1 + d_theta2 * strength2
                new_d_phi = d_phi1 * strength1 + d_phi2 * strength2
                new_d_r = (d_r1 - 1.0) * strength1 + (d_r2 - 1.0) * strength2 + 1.0
            else:
                # 如果运镜类型不存在，使用默认静态参数
                raise ValueError(f"Unknown motion type: {motion_type1} or {motion_type2}")

            # 构建8位参数格式: [start_frame_index, end_frame_index, x_offset, y_offset, z_offset, d_theta, d_phi, d_r]
            new_info = [
                start_frame_index,
                end_frame_index,
                new_x_offset,
                new_y_offset,
                new_z_offset,
                new_d_theta,
                new_d_phi,
                new_d_r,
            ]
            new_camera_control_info.append(new_info)

        else:
            raise ValueError(f"Unknown camera control info length: {len(info), info}")

    return new_camera_control_info, is_swing
