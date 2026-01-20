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

    input_data["cond_audio"]["person1"] = audio_emb_path_list[0]

    if "bbox" in input_data:
        if type(input_data["bbox"]) == dict:
            bboxes = sorted(
                input_data["bbox"].items(),
                key=lambda item: int(item[0].replace("person", "")),
            )
            input_data["bbox"] = [box for (key, box) in bboxes]

        assert len(input_data["bbox"]) == len(input_data["cond_audio"])

    if len(input_data["cond_audio"]) > 1:
        silent_speech = np.zeros(sum_human_speechs.shape[0])
        silent_audio_embedding = get_embedding(silent_speech, wav2vec_feature_extractor, audio_encoder)
        silent_emb_path = os.path.join(audio_save_dir, "silent.pt")
        torch.save(silent_audio_embedding, silent_emb_path)
        input_data["silent_audio_embs"] = silent_emb_path
    else:
        input_data["silent_audio_embs"] = None

    _ext_info["final_video_length"] = trans_list[-1]

    print(f"preprocess audio time: {time.time()-start:0.2f}s")
    return input_data, _ext_info
