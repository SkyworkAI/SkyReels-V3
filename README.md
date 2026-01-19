<p align="center">
  <img src="assets/logo2.png" alt="SkyReels Logo" width="50%">
</p>

<h1 align="center">SkyReels V3: Multimodal Video Generation Model</h1> 

<p align="center">
👋 <a href="https://www.skyreels.ai/" target="_blank">Playground</a> · 🤗 <a href="https://huggingface.co/collections/Skywork/skyreels-v2-6801b1b93df627d441d0d0d9" target="_blank">Hugging Face</a> · 🤖 <a href="https://www.modelscope.cn/collections/SkyReels-V2-f665650130b144" target="_blank">ModelScope</a>
</p>

---
Welcome to the **SkyReels V3** repository! This is the official release of our flagship video generation model, built upon a unified **multimodal in-context learning framework**. SkyReels V3 natively supports three core generative capabilities: **1) multi-subject video generation from reference images**, **2) video generation guided by audio**, and **3) video-to-video generation**.

We also provide API access to this model. You can integrate and use the SkyReels V3 series models through the **[SkyReels Developer Platform](https://www.skyreels.ai/)**.

## 🔥🔥🔥 News!!
* Jan 21, 2026: 🎉 We release the inference code and model weights of [SkyReels-V3](https://github.com/SkyworkAI/SkyReels-V3).
* Jun 1, 2025: 🎉 We published the technical report, [SkyReels-Audio: Omni Audio-Conditioned Talking Portraits in Video Diffusion Transformers](https://arxiv.org/pdf/2506.00830).
* May 16, 2025: 🔥 We release the inference code for [video extension](#ve) and [start/end frame control](#se) in diffusion forcing model.
* Apr 24, 2025: 🔥 We release the 720P models, [SkyReels-V2-DF-14B-720P](https://huggingface.co/Skywork/SkyReels-V2-DF-14B-720P) and [SkyReels-V2-I2V-14B-720P](https://huggingface.co/Skywork/SkyReels-V2-I2V-14B-720P). The former facilitates infinite-length autoregressive video generation, and the latter focuses on Image2Video synthesis.
* Apr 21, 2025: 👋 We release the inference code and model weights of [SkyReels-V2](https://huggingface.co/collections/Skywork/skyreels-v2-6801b1b93df627d441d0d0d9) Series Models and the video captioning model [SkyCaptioner-V1](https://huggingface.co/Skywork/SkyCaptioner-V1) .
* Apr 3, 2025: 🔥 We also release [SkyReels-A2](https://github.com/SkyworkAI/SkyReels-A2). This is an open-sourced controllable video generation framework capable of assembling arbitrary visual elements.
* Feb 18, 2025: 🔥 we released [SkyReels-A1](https://github.com/SkyworkAI/SkyReels-A1). This is an open-sourced and effective framework for portrait image animation.
* Feb 18, 2025: 🔥 We released [SkyReels-V1](https://github.com/SkyworkAI/SkyReels-V1). This is the first and most advanced open-source human-centric video foundation model.

## 🎥 Demos
<table>
  <tr>
    <td align="center" width="33%">
      <a href="https://www.skyreels.ai/videos/ReferencetoVideo/1.mp4">
        <img src="assets/ref_to_video.gif" width="100%" alt="Reference to Video">
      </a>
      <br><b>Reference to Video</b>
    </td>
    <td align="center" width="33%">
      <a href="https://www.skyreels.ai/videos/VideoExtension/1.mp4">
        <img src="assets/video_ext.gif" width="100%" alt="Video Extension">
      </a>
      <br><b>Video Extension</b>
    </td>
    <td align="center" width="33%">
      <a href="https://www.skyreels.ai/videos/TalkingAvatar/1.mp4">
        <img src="assets/talking_avatar.gif" width="100%" alt="Talking Avatar">
      </a>
      <br><b>Talking Avatar</b>
    </td>
  </tr>
</table>

The demos above showcase videos generated using our SkyReels-V3 unified multimodal in-context learning framework.

## 🚀 Quickstart

#### Installation
```shell
# clone the repository.
git clone https://github.com/SkyworkAI/SkyReels-V3
cd SkyReels-V3
# Install dependencies. Test environment uses Python 3.10.12.
pip install -r requirements.txt
```

#### Model Download
You can download our models from Hugging Face:
<table>
  <thead>
    <tr>
      <th>Type</th>
      <th>Variant</th>
      <th>Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Reference to Video</td>
      <td>14B-720P</td>
      <td>🤗 <a href="#">Huggingface</a> 🤖 <a href="#">ModelScope</a></td>
    </tr>
    <tr>
      <td>Video Extension</td>
      <td>14B-720P</td>
      <td>🤗 <a href="#">Huggingface</a> 🤖 <a href="#">ModelScope</a></td>
    </tr>
    <tr>
      <td>Talking Avatar</td>
      <td>14B-720P</td>
      <td>🤗 <a href="#">Huggingface</a> 🤖 <a href="#">ModelScope</a></td>
    </tr>
  </tbody>
</table>

After downloading, set the model path in your generation commands:

#### Reference to Video
Reference-to-Video is a model that synthesizes coherent video sequences from 1 to 4 reference images and a text prompt. It excels at maintaining strong identity fidelity and narrative consistency for characters, objects, and backgrounds.
- Single-GPU inference
```bash
python3 generate_video.py --task_type reference_to_video --ref_imgs "https://skyreels-api.oss-accelerate.aliyuncs.com/examples/subject_reference/0_1.png,https://skyreels-api.oss-accelerate.aliyuncs.com/examples/subject_reference/0_2.png" --prompt "two girls talking in a club" --duration 5 --offload
```
- Multi-GPU inference using xDiT USP
```bash
torchrun --nproc_per_node=4 generate_video.py --task_type reference_to_video --ref_imgs "https://skyreels-api.oss-accelerate.aliyuncs.com/examples/subject_reference/0_1.png,https://skyreels-api.oss-accelerate.aliyuncs.com/examples/subject_reference/0_2.png" --prompt "two girls talking in a club" --duration 5 --use_usp
```
> 💡Note: 
> * The ***task_type*** parameter must be set to "reference_to_video".
> * The ***ref_imgs*** parameter accepts 1 to 4 reference images. When providing multiple images, please separate their paths or URLs with commas.
> * The recommended output specification for this model is a 5-second video at 720p and 24 fps.

#### Video Extension
Video Extension is a suite of models designed to extend existing videos while preserving motion continuity, scene coherence, and the visual identity of subjects. It includes two main models: Single-shot Video Extension and Shot Switching Video Extension.

- **Single-shot Video Extension** supports video extension from 5 seconds to 30 seconds.

- **Shot Switching Video Extension** is designed for video extension with specified shot transitions, supporting cinematography types such as "Cut-In", "Cut-Out", "Shot/Reverse Shot", "Multi-Angle", and "Cut Away", but is currently limited to 5-second extensions.

##### Single-shot Video Extension
- Single-GPU inference
```bash
python3 generate_video.py --task_type single_shot_extension --input_video https://skyreels-api.oss-accelerate.aliyuncs.com/examples/video_extension/test.mp4 --prompt "A man is making his way forward slowly, leaning on a white cane to prop himself up." --duration 5 --offload
```
- Multi-GPU inference using xDiT USP
```bash
torchrun --nproc_per_node=4 generate_video.py --task_type single_shot_extension --input_video https://skyreels-api.oss-accelerate.aliyuncs.com/examples/video_extension/test.mp4 --prompt "A man is making his way forward slowly, leaning on a white cane to prop himself up." --duration 5 --offload --use_usp
```
> 💡Note: 
> * The ***task_type*** parameter must be set to "single_shot_extension".
> * The **input_video** parameter specifies the source video to be extended. Since the **single_shot_extension** model supports extensions of 5 to 30 seconds, the **duration** parameter accepts an integer value within this range.
##### Shot Switching Video Extension
- Single-GPU inference
```bash
python3 generate_video.py --task_type shot_switching_extension --input_video https://skyreels-api.oss-cn-hongkong.aliyuncs.com/examples/video_extension/6.mp4 --prompt "Create a top side angle view of the robot playing the guitar" --offload
```
- Multi-GPU inference using xDiT USP
```bash
torchrun --nproc_per_node=4 generate_video.py --task_type shot_switching_extension --input_video https://skyreels-api.oss-cn-hongkong.aliyuncs.com/examples/video_extension/6.mp4 --prompt "Create a top side angle view of the robot playing the guitar" --offload --use_usp
```
> 💡Note: 
> * The ***task_type*** parameter must be set to "shot_switching_extension".
> * The **input_video** parameter specifies the source video to be extended, and the **duration** parameter is therefore limited to a maximum of 5 seconds.
> * The model supports various cinematography transitions such as "Cut-In" and "Cut-Away". For optimal output, consider using an LLM to integrate these techniques into well-structured generation prompts.

#### Talking Avatar
The Talking Avatar model generates vibrant, lifelike talking avatars from a single portrait image and an audio clip, supporting videos of up to 200 seconds in length. It is capable of producing multi-avatar scenes, adapting to diverse artistic styles, and delivering performances with rich expressiveness and precise synchronization.
- Single-GPU inference
```bash
```
- Multi-GPU inference using xDiT USP
```bash
```