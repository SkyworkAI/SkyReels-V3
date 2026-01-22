<p align="center">
  <img src="assets/logo2.png" alt="SkyReels Logo" width="50%">
</p>

<h1 align="center">SkyReels V3: Multimodal Video Generation Model</h1> 

<p align="center">
👋 <a href="https://www.skyreels.ai/" target="_blank">Playground</a> · 🤗 <a href="https://huggingface.co/collections/Skywork/skyreels-v3" target="_blank">Hugging Face</a> · 🤖 <a href="https://www.modelscope.cn/collections/Skywork/SkyReels-V3" target="_blank">ModelScope</a> · 🌐 <a href="https://github.com/SkyworkAI/SkyReels-V3" target="_blank">GitHub</a>
</p>

---
Welcome to the **SkyReels V3** repository! This is the official release of our flagship video generation model, built upon a unified **multimodal in-context learning framework**. SkyReels V3 natively supports three core generative capabilities: **1) multi-subject video generation from reference images**, **2) video generation guided by audio**, and **3) video-to-video generation**.

We also provide API access to this model. You can integrate and use the SkyReels V3 series models through the **[SkyReels Developer Platform](https://platform.skyreels.ai/)**.

## 🔥🔥🔥 News!!
* Jan 21, 2026: 🎉 We release the inference code and model weights of [SkyReels-V3](https://github.com/SkyworkAI/SkyReels-V3).
* Dec 1, 2025: 🎉 We launched the API for the SkyReels-V3 models on the [SkyReels Developer Platform](https://platform.skyreels.ai/).
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
        <img src="https://skyreels-api.oss-cn-hongkong.aliyuncs.com/examples/ref_to_video.gif" width="100%" alt="Reference to Video">
      </a>
      <br><b>Reference to Video</b>
    </td>
    <td align="center" width="33%">
      <a href="https://www.skyreels.ai/videos/VideoExtension/1.mp4">
        <img src="https://skyreels-api.oss-cn-hongkong.aliyuncs.com/examples/video_ext.gif" width="100%" alt="Video Extension">
      </a>
      <br><b>Video Extension</b>
    </td>
    <td align="center" width="33%">
      <a href="https://www.skyreels.ai/videos/TalkingAvatar/1.mp4">
        <img src="https://skyreels-api.oss-cn-hongkong.aliyuncs.com/examples/talking_avatar.gif" width="100%" alt="Talking Avatar">
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
# Install dependencies. Test environment uses Python 3.12.3 & cuda 12.8.
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
      <td>🤗 <a href="https://huggingface.co/Skywork/SkyReels-V3-R2V-14B">Huggingface</a> 🤖 <a href="https://www.modelscope.cn/models/Skywork/SkyReels-V3-R2V-14B">ModelScope</a></td>
    </tr>
    <tr>
      <td>Video Extension</td>
      <td>14B-720P</td>
      <td>🤗 <a href="https://huggingface.co/Skywork/SkyReels-V3-V2V-14B">Huggingface</a> 🤖 <a href="https://www.modelscope.cn/models/Skywork/SkyReels-V3-V2V-14B">ModelScope</a></td>
    </tr>
    <tr>
      <td>Talking Avatar</td>
      <td>19B-720P</td>
      <td>🤗 <a href="https://huggingface.co/Skywork/SkyReels-V3-A2V-19B">Huggingface</a> 🤖 <a href="https://www.modelscope.cn/models/Skywork/SkyReels-V3-A2V-19B">ModelScope</a></td>
    </tr>
  </tbody>
</table>

After downloading, set the model path in your generation commands:

#### Reference to Video
Reference-to-Video is a model that synthesizes coherent video sequences from 1 to 4 reference images and a text prompt. It excels at maintaining strong identity fidelity and narrative consistency for characters, objects, and backgrounds.
- Single-GPU inference
```bash
python3 generate_video.py --task_type reference_to_video --ref_imgs "https://skyreels-api.oss-accelerate.aliyuncs.com/examples/subject_reference/0_1.png,https://skyreels-api.oss-accelerate.aliyuncs.com/examples/subject_reference/0_2.png" --prompt "In a dimly lit, cluttered occult club room adorned with shelves full of books, skulls, and mysterious dolls, two young Asian girls are talking. One girl has vibrant teal pigtails with bangs, wearing a white collared polo shirt, while the other has a sleek black bob with bangs, also in a white polo shirt, conversing under the hum of fluorescent lights, a high-quality and detailed cinematic shot." --duration 5 --offload
```
- Multi-GPU inference using xDiT USP
```bash
torchrun --nproc_per_node=4 generate_video.py --task_type reference_to_video --ref_imgs "https://skyreels-api.oss-accelerate.aliyuncs.com/examples/subject_reference/0_1.png,https://skyreels-api.oss-accelerate.aliyuncs.com/examples/subject_reference/0_2.png" --prompt "In a dimly lit, cluttered occult club room adorned with shelves full of books, skulls, and mysterious dolls, two young Asian girls are talking. One girl has vibrant teal pigtails with bangs, wearing a white collared polo shirt, while the other has a sleek black bob with bangs, also in a white polo shirt, conversing under the hum of fluorescent lights, a high-quality and detailed cinematic shot." --duration 5 --offload --use_usp
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
python3 generate_video.py --task_type shot_switching_extension --input_video https://skyreels-api.oss-accelerate.aliyuncs.com/examples/video_extension/test.mp4 --prompt "[ZOOM_IN_CUT] The scene cuts from a medium shot of a visually impaired man walking on a path in a park. The shot then cut in to a close-up of the man's face and upper torso. The visually impaired Black man is shown from the chest up, wearing dark sunglasses, a grey turtleneck scarf, and a light olive green jacket. His head is held straight, looking forward towards the camera, continuing his walk. The lighting is natural and bright. The background is a soft blur of green trees and foliage from the park." --offload
```
- Multi-GPU inference using xDiT USP
```bash
torchrun --nproc_per_node=4 generate_video.py --task_type shot_switching_extension --input_video https://skyreels-api.oss-accelerate.aliyuncs.com/examples/video_extension/test.mp4 --prompt "[ZOOM_IN_CUT] The scene cuts from a medium shot of a visually impaired man walking on a path in a park. The shot then cut in to a close-up of the man's face and upper torso. The visually impaired Black man is shown from the chest up, wearing dark sunglasses, a grey turtleneck scarf, and a light olive green jacket. His head is held straight, looking forward towards the camera, continuing his walk. The lighting is natural and bright. The background is a soft blur of green trees and foliage from the park." --offload --use_usp

```
> 💡Note: 
> * The ***task_type*** parameter must be set to "shot_switching_extension".
> * The **input_video** parameter specifies the source video to be extended, and the **duration** parameter is therefore limited to a maximum of 5 seconds.
> * To effectively utilize the supported cinematography types ("Cut-In", "Cut-Out", "Shot/Reverse Shot", "Multi-Angle", "Cut Away"), you can use a Large Language Model (LLM) to craft and optimize your generation prompts, ensuring clear and precise creative direction.

#### Talking Avatar
The Talking Avatar model generates vibrant, lifelike talking avatars from a single portrait image and an audio clip, supporting videos of up to 200 seconds in length. It is capable of producing multi-avatar scenes, adapting to diverse artistic styles, and delivering performances with rich expressiveness and precise synchronization.
- Single-GPU inference
```bash
python3 generate_video.py --task_type talking_avatar --prompt "A woman is giving a speech. She is confident, poised, and joyful. Use a static shot." --seed 42 --offload --input_image "https://skyreels-api.oss-accelerate.aliyuncs.com/examples/talking_avatar_video/woman.JPEG" --input_audio "https://skyreels-api.oss-accelerate.aliyuncs.com/examples/talking_avatar_video/single_actor/woman_speech.mp3"
```
- Multi-GPU inference using xDiT USP
```bash
torchrun --nproc_per_node=4 generate_video.py --task_type talking_avatar --prompt "A woman is giving a speech. She is confident, poised, and joyful. Use a static shot." --seed 42 --use_usp --offload --input_image "https://skyreels-api.oss-accelerate.aliyuncs.com/examples/talking_avatar_video/woman.JPEG" --input_audio "https://skyreels-api.oss-accelerate.aliyuncs.com/examples/talking_avatar_video/single_actor/woman_speech.mp3"
```
> 💡Note: 
> * The ***task_type*** parameter must be set to "talking_avatar".
> * The **input_image** parameter specifies the first-frame image for talking avatar generation (URL or local path). Supported formats: **jpg/jpeg, png, gif, bmp**.
> * The **input_audio** parameter specifies the driving audio (URL or local path). Currently supports **1 audio**. Supported formats: **mp3, wav**. Audio duration: **<= 200 seconds**.

## Introduction of SkyReels-V3

### Reference to Video
SkyReels-V3 Multi-Reference Video Generation Model is a new-generation video synthesis system independently developed by SkyReels. The model enables users to input 1 to 4 reference images—including character portraits, object images, and background scenes—and generates coherent video sequences aligned with textual instructions, ensuring logical compositional relationships and narrative progression. With robust capabilities in dynamic scene generation, the model is widely applicable across various domains such as video production, social media entertainment, live-stream commerce, and product demonstration.


> Key Features :
> * Supports fusion of up to 4 reference images, including character, object, and background references.
> * Exceptional subject consistency and composition coherence, with industry-leading motion generation quality.
> * Multiple aspect ratios: 1:1, 3:4, 4:3, 16:9, 9:16.

#### Model Overview
The model achieves high subject and background consistency while accurately responding to user instructions. To enhance its capability of preserving reference image content, the SkyReels team developed a comprehensive data processing pipeline. This pipeline employs a cross-frame pairing strategy to select reference frames from continuous video sequences and utilizes image editing models to extract subject images, simultaneously accomplishing background completion and semantic rewriting—effectively avoiding the "copy-paste" effect.

During the training phase, the SkyReels team introduced an image-video hybrid training mechanism and supported multi-resolution joint training, significantly improving the model's generalization performance. Evolving from the SkyReels V2 to the V3 version, the model has reached the level of industry-leading closed-source SOTA (state-of-the-art) models across multiple evaluation metrics, demonstrating top-tier comprehensive generation capabilities in the field.

#### 📊 Performance Comparison

| Model | Reference Consistency ↑ | Instruction Following ↑ | Visual Quality ↑ |
|-------|-------------------------|-------------------------|------------------|
| Vidu Q2 | 0.5961 | 27.84 | 0.7877 |
| Kling 1.6 | 0.6630 | 29.23 | 0.8034 |
| PixVerse V5 | 0.6542 | 29.34 | 0.7976 |
| **SkyReels V3** | **0.6698** | **27.22** | **0.8119** |

### Video Extension

SkyReels-V3 Video Extension Model is a new-generation video generation system independently developed by SkyReels. The model allows users to input an existing video segment and extend it with coherent, logically consistent subsequent scenes based on textual instructions. It is widely applicable in scenarios such as video production, short-form series creation, live commerce, and product demonstration.

> Key Features :
> * Dual Extension Modes: Supports both single-shot continuation and multi-shot switching (with 5 transition types), operable via manual selection or automatic detection.
> * Superior Visual Quality: Excellent aesthetic composition, robust motion quality, and seamless continuity preservation.
> * Outstanding Style Adherence: Strictly follows input visual styles (realistic, cinematic, or specialized aesthetics) with exceptional compatibility.
> * High-Definition Output: Ensures premium content quality, supporting 720P resolution.
> * Flexible Duration Control: Adjustable output length between 5 to 30 seconds for sing-shot video extension.
> * Customizable Aspect Ratios: Supports multiple ratios including 1:1, 3:4, 4:3, 16:9, and 9:16.

#### Model Overview
The SkyReels-V3 Video Extension Engine deeply integrates spatiotemporal consistency modeling with large-scale video understanding, breaking through the frame-level limitations of traditional video generation to achieve a qualitative leap from "visual continuation" to "narrative continuation." As the industry's first engine supporting intelligent shot switching during video extension, SkyReels-V3 not only achieves top-tier temporal coherence but also extends generation capacity to minute-level durations through an innovative history enhancement mechanism, ensuring depth and stability in long-form video storytelling. 

The engine accurately parses scene semantics, motion trajectories, and emotional context from the original video, while intelligently planning the composition, character behavior, and cinematography of the extended content. It supports both seamless single-shot continuation and multi-type shot switching—including professional techniques such as Cut-In, Cut-Out, Reverse Shot, Multi-Angle, and Cut Away—automatically generating extended clips with strong narrative logic and visual coherence. This empowers visual language with cinematic dynamism and tension, marking a true generational shift from frame interpolation to plot creation.

Technical Innovations:
- Unified multi-segment positional encoding and hybrid hierarchical data training enable precise motion prediction and smooth transitions in complex scenes.
- A novel fusion of token concatenation flexibility and channel concatenation efficiency significantly reduces total token count while enhancing computational performance without compromising output quality.
- The architecture robustly handles challenges such as rapid motion, multi-person interactions, and abrupt scene changes, strictly ensuring physical plausibility and emotional consistency.
- In intelligent shot switching, the system dynamically plans cut rhythms and viewpoint variations based on video semantics and user prompts, generating freely lengthened, professionally shot-extended content within a unified style.

With outstanding generalization capabilities, SkyReels-V3 achieves state-of-the-art (SOTA) performance on core metrics including single-shot and multi-shot extension. It is widely adaptable to diverse scenarios such as live-action filmmaking, short-series industrial production, game cinematics, and security footage enhancement. The generated content delivers high-definition visuals, sharp details, and natural motion fluency, offering professional creators a "what-you-see-is-what-you-think" extension experience and redefining the boundaries of video generation.

### Talking Avatar

Create with just one image and audio clip.

> Key Features :
> * Superior visual quality and precise lip sync. Generate 720p HD videos at 24 fps for smooth and clear results. Supports multiple languages to ensure lip movements match the audio, enhancing authenticity.
> * Multi-style support. Compatible with real-life, cartoon, animal, and stylized characters—offering creative flexibility for brand ambassadors or virtual IPs.
> * Long-form video generation. Produce minute-long coherent videos for detailed explanations, news reports, training courses, and more.
> * Multi-character scenes. Optimized for group interactions, allowing role assignments to support dialogues, interviews, and other dynamic content.

#### Model Overview

Powered by advanced multimodal understanding techniques, SkyReels Avatars don’t just “hear sound”—they truly understand your content. By analyzing voice, image, and emotional cues, they generate expressions, movements, and camera language that naturally align with your intent.
Built on a scalable diffusion Transformer architecture and trained with audio-visual alignment strategies, our technology ensures highly accurate lip sync. Whether it’s Chinese, English, Korean, singing, or fast-paced dialogue—the lip movements match the pronunciation for a realistic audiovisual experience.

Using a keyframe-constrained generation framework, the model first structures key content before smoothly connecting transitions. This ensures consistent character appearance and fluid motion, even in long videos. Generate high-quality minute-long videos in one go—ideal for explanations, broadcasts, storytelling, and more.
From real people and anime characters to pets and artwork—any image can be turned into a lifelike digital avatar.

In internal evaluations against mainstream avatar models, SkyReels model excel across multiple dimensions—overall quality, lip sync, and expressiveness—achieving a significantly higher overall rating.

#### 📊 Performance Comparison

| Model | Audio-Visual Sync ↑ | Visual Quality ↑ | Charactr Consistency ↑ |
|-------|-------------------------|-------------------------|------------------|
| OmniHuman 1.5 | **8.25** | 4.60 | **0.81** |
| KlingAvatar | 8.01 | 4.55 | 0.78 |
| HunyuanAvatar | 6.72 | 4.50 | 0.74 |
| **SkyReels V3** | 8.18 | **4.60** | 0.80 |

## Acknowledgements
We would like to thank the contributors of <a href="https://github.com/Wan-Video/Wan2.1">Wan 2.1</a>, <a href="https://github.com/MeiGen-AI/MultiTalk">MultiTalk</a>, <a href="https://github.com/xdit-project/xDiT">XDit</a> and <a href="https://github.com/huggingface/diffusers">diffusers</a> repositories, for their open research and contributions.

## Github Star History

<a href="https://star-history.com/#SkyworkAI/SkyReels-V3&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=SkyworkAI/SkyReels-V3&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=SkyworkAI/SkyReels-V3&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=SkyworkAI/SkyReels-V3&type=Date" />
 </picture>
</a>