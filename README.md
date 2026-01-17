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

#### Video Extension

#### Talking Avatar