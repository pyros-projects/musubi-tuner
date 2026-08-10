# Musubi Tuner

[English](./README.md) | [日本語](./README.ja.md)

> [!NOTE]
> **This fork adds MiniMax H3 image-LoRA training.** The quick start is right below; everything after it is the regular Musubi Tuner README (installation lives there). Full H3 details: [docs/minimax_h3.md](./docs/minimax_h3.md).

## MiniMax H3 Image LoRA — Quick Start

Train a character or style LoRA for [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (the 33B video+audio model) **from still images only**, on a single consumer GPU, using the same INT8 ConvRot model files your ComfyUI H3 setup already uses. Images go in, a normal `.safetensors` LoRA comes out that works for both H3 image generation and video.

Image-only training on a video model normally wrecks motion: jump cuts, melting fur, slideshow vibes. The defaults in this fork encode everything we found while debugging exactly that:

- **`no_packed_attn` LoRA preset.** H3 is a single-stream DiT — there are no separate "temporal blocks" you could exclude, because temporal mixing happens inside the same packed attention as everything else. So the default preset skips attention entirely: LoRA weights go only on the per-token MLPs and the text token refiner. MLPs act on each token independently and physically cannot move information between frames — likeness trains, motion stays stock. (Presets that also train attention, like `attn_mlp`, give slightly stronger stills and visibly broken video.)
- **Resolution-shifted timesteps instead of H3's video schedule.** H3's native shift-12 schedule spends most training steps at very high noise, where video layout and motion live — an image LoRA trained there learns the least likeness while doing the most temporal damage. The `image` preset (`krea2_shift`) instead draws timesteps from a resolution-dependent distribution centered where image identity actually lives.
- **`max_timestep 875` cutoff.** The very top of the noise range is composition-and-motion territory. Capping training below it (`preserve_distribution_shape` keeps the rest of the curve unchanged) was the single biggest "stills stay good, video stops degrading" lever in our A/B runs.
- **Two-latent preview decode.** H3's ViT video decoder cannot decode a lone single-frame latent — it's out of distribution and comes out soft. Training previews instead sample a duplicated 2-frame packet with a 2nd-order (AB2) solver and decode the last frame — measured ~8 dB PSNR better than the naive single-latent decode.

### You need

- A working ComfyUI install that already runs MiniMax H3 — the trainer borrows its Python env to run the INT8 text encoder — including the three model files you already have for it:
  - DiT: `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
  - Text encoder: `qwen3vl_32b_minimax_h3_int8_convrot.safetensors`
  - VAE: `minimax_h3_video_vae_fp16.safetensors`
- This repo installed with its own venv ([Installation](#installation) below).
- A folder of images of your subject with `.txt` captions next to them. Captioning rule: **describe everything you want to keep control over.** If every photo has a white door behind your cat and you never mention it, the white door becomes part of your cat.

### 1. Point train.sh at your setup

Open `.pyro/h3/train.sh` and edit the paths at the top: `MUSUBI_VENV` (this repo's venv), `COMFYUI_DIR`, `DIT`, `TEXT_ENCODER`, `VAE` — and `OUTPUT_DIR` further down (where LoRAs get written).

### 2. Create a dataset file

Copy the demo `.pyro/h3/cfg/lucy_v2.toml` to `.pyro/h3/cfg/<yourname>.toml` and set `image_directory` (your images + captions) and `cache_directory` (anywhere writable — latents get cached there once):

```toml
[general]
resolution = [768, 768]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = true

[[datasets]]
image_directory = "/path/to/your/images"
cache_directory = "/path/to/cache/yourname"
num_repeats = 1
```

### 3. Create a prompt file

Copy `.pyro/h3/cfg/p_lucy_v2.toml` to `.pyro/h3/cfg/p_<yourname>.toml` and put your trigger prompt in it. These render as preview images every 50 steps during training, so you watch the LoRA learn instead of praying. `frame_count = 1` is a still; any `frame_count` of the form 17n+5 (22, 39, 73, …) renders a short silent video preview instead.

### 4. Train

```bash
# first run: caches latents + text embeddings, then trains
H3_NAME=yourname CACHE_DATASET=1 .pyro/h3/train.sh

# later runs: reuse the cache
H3_NAME=yourname .pyro/h3/train.sh
```

Checkpoints land in `OUTPUT_DIR` every 100 steps as regular LoRA `.safetensors` — load them in ComfyUI like any other LoRA. Each save also writes a `.comfy.safetensors` twin in ComfyUI/ai-toolkit key format for tools that expect `diffusion_model.*.lora_A/lora_B` keys.

### Tips from our testing

- **Inference strength ~0.8** is the sweet spot. 1.0 gives maximum likeness but can mangle fast motion (jumps, landings); 0.5 loses identity.
- **Judge results with EasyCache off.** Cache reuse adds background wobble that is not your LoRA's fault and dampens real motion. It scammed us for a day.
- With the default Prodigy optimizer, 500–1000 steps is usually plenty at rank 32.
- Tight on VRAM? Raise `BLOCKS_TO_SWAP` (default 6) — slower but smaller.
- Slow previews? Point `SAMPLE_LORA_OVERLAY` at the community [Turbo LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) and set `sample_steps = 4` in your prompt file — ~5× faster preview ticks, applied only during sampling, never trained into your LoRA. Keep one clean 20-step prompt (`sample_lora_overlay = 0`) for judging.
- Before publishing weights, read the MiniMax H3 community license — it has jurisdiction restrictions.

---

## Table of Contents

<details>
<summary>Click to expand</summary>

- [Musubi Tuner](#musubi-tuner)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
    - [Sponsors](#sponsors)
    - [Support the Project](#support-the-project)
    - [Recent Updates](#recent-updates)
    - [Releases](#releases)
    - [For Developers Using AI Coding Agents](#for-developers-using-ai-coding-agents)
  - [Overview](#overview)
    - [Hardware Requirements](#hardware-requirements)
    - [Features](#features)
    - [Documentation](#documentation)
  - [Installation](#installation)
    - [pip based installation](#pip-based-installation)
    - [uv based installation](#uv-based-installation-experimental)
    - [Linux/MacOS](#linuxmacos)
    - [Windows](#windows)
  - [Model Download](#model-download)
  - [Usage](#usage)
    - [Dataset Configuration](#dataset-configuration)
    - [Pre-caching and Training](#pre-caching-and-training)
    - [Configuration of Accelerate](#configuration-of-accelerate)
    - [Training and Inference](#training-and-inference)
  - [Miscellaneous](#miscellaneous)
    - [SageAttention Installation](#sageattention-installation)
    - [PyTorch version](#pytorch-version)
  - [Disclaimer](#disclaimer)
  - [Contributing](#contributing)
  - [License](#license)

</details>

## Introduction

This repository provides scripts for training LoRA (Low-Rank Adaptation) models with HunyuanVideo, Wan2.1/2.2, FramePack, FLUX.1 Kontext, FLUX.2 dev/klein, Qwen-Image, Z-Image, and MiniMax H3 architectures.

This repository is unofficial and not affiliated with the official repositories of these architectures.

*This repository is under development.*

### Sponsors

We are grateful to the following companies for their generous sponsorship:

<a href="https://aihub.co.jp/top-en">
  <img src="./images/logo_aihub.png" alt="AiHUB Inc." title="AiHUB Inc." height="100px">
</a>

### Support the Project

If you find this project helpful, please consider supporting its development via [GitHub Sponsors](https://github.com/sponsors/kohya-ss/). Your support is greatly appreciated!

### Recent Updates

- August 2, 2026
    - Added experimental MiniMax H3 FL2VA joint video/audio LoRA training support. See the [MiniMax H3 documentation](./docs/minimax_h3.md).

GitHub Discussions Enabled: We've enabled GitHub Discussions for community Q&A, knowledge sharing, and technical information exchange. Please use Issues for bug reports and feature requests, and Discussions for questions and sharing experiences. [Join the conversation →](https://github.com/kohya-ss/musubi-tuner/discussions)

- June 24, 2026
    - Added experimental support for Krea 2 (LoRA training and inference). See [PR #980](https://github.com/kohya-ss/musubi-tuner/pull/980) for details.
        - For details, please refer to the [documentation](./docs/krea2.md).

- June 19, 2026
    - Added experimental support for Ideogram4 (LoRA training and inference). Many thanks to sdbds for [PR #966](https://github.com/kohya-ss/musubi-tuner/pull/966). Follow-ups were made in [PR #975](https://github.com/kohya-ss/musubi-tuner/pull/975) and [PR #977](https://github.com/kohya-ss/musubi-tuner/pull/977). Please refer to the PRs for detailed changes.
        - For details, please refer to the [documentation](./docs/ideogram4.md).
        - JSON format prompts are recommended, but natural language training is also possible.
        - Training settings details are unknown, so community information sharing is welcome.

- June 16, 2026
    - Added H2D-only block swap, an optimized block swap mode for LoRA (LoHa/LoKr) training, available for all architectures. Enable it with `--block_swap_h2d_only`. See [PR #972](https://github.com/kohya-ss/musubi-tuner/pull/972).
        - For frozen-base training, the base weights on the CPU and GPU are identical, so the classic block swap's device-to-host (D2H) copy is pure overhead. H2D-only keeps a permanent master copy on the CPU and only ever transfers host-to-device, removing the D2H transfer entirely. This can improve training throughput, with the largest benefit when using `--fp8_base` / `--fp8_scaled`.
        - Requires `--gradient_checkpointing`. The number of GPU ring buffers used for streaming can be tuned with `--block_swap_ring_size` (default `2`; `1` minimizes VRAM).
        - There is also a new standalone [Block Swap documentation](./docs/block_swap.md) covering all block swap options.

- June 13, 2026
    - Added the `--save_precision` option for network weights and changed the default save precision to fp32. Thank you rockerBOO [PR #967](https://github.com/kohya-ss/musubi-tuner/pull/967).
        - **Breaking Change**: The default save precision for LoRA files has been changed to fp32.
        - This preserves the precision of LoRA weights during training and is useful for post-processing such as post-hoc EMA, merging, extraction, and weight analysis.
        - LoRA files may be about twice as large as before when training with `--mixed_precision bf16`(`fp16`). To keep the previous behavior, specify `--save_precision bf16` (`fp16`).
        - Please refer to the [HunyuanVideo documentation](./docs/hunyuan_video.md#training--学習) for details.

- June 8, 2026
    - Added experimental support for HiDream-O1-Image (LoRA training, full finetuning, and inference). See [PR #964](https://github.com/kohya-ss/musubi-tuner/pull/964).
        - Please refer to the [documentation](./docs/hidream_o1.md) for details.
        - An optional DINOv3 auxiliary perceptual loss is also available. See the [advanced configuration documentation](./docs/advanced_config.md).
        - Many thanks to sdbds for [PR #947](https://github.com/kohya-ss/musubi-tuner/pull/947) (followed by [PR #955](https://github.com/kohya-ss/musubi-tuner/pull/955)), which this support is based on. Please open the PRs if you would like to review the changes in detail.

- May 22, 2026
    - Performed a large-scale internal refactoring to improve code quality and maintainability. See [PR #950](https://github.com/kohya-ss/musubi-tuner/pull/950)
        - We have taken care to ensure that there are no direct impacts on users. For details and to report any issues, please refer to [this discussion](https://github.com/kohya-ss/musubi-tuner/discussions/949).

### Releases

We are grateful to everyone who has been contributing to the Musubi Tuner ecosystem through documentation and third-party tools. To support these valuable contributions, we recommend working with our [releases](https://github.com/kohya-ss/musubi-tuner/releases) as stable reference points, as this project is under active development and breaking changes may occur.

You can find the latest release and version history in our [releases page](https://github.com/kohya-ss/musubi-tuner/releases).

### For Developers Using AI Coding Agents

This repository provides recommended instructions to help AI agents like Claude and Gemini understand our project context and coding standards.

To use them, you need to opt-in by creating your own configuration file in the project root.

**Quick Setup:**

1.  Create a `CLAUDE.md`, `GEMINI.md`, and/or `AGENTS.md` file in the project root.
2.  Add the following line to your `CLAUDE.md` to import the repository's recommended prompt (currently they are the almost same):

    ```markdown
    @./.ai/claude.prompt.md
    ```

    or for Gemini:

    ```markdown
    @./.ai/gemini.prompt.md
    ```

    You may be also import the prompt depending on the agent you are using with the custom prompt file such as `AGENTS.md`.

3.  You can now add your own personal instructions below the import line (e.g., `Always include a short summary of the change before diving into details.`).

This approach ensures that you have full control over the instructions given to your agent while benefiting from the shared project context. Your `CLAUDE.md`, `GEMINI.md` and `AGENTS.md` (as well as Claude's `.mcp.json`) are already listed in `.gitignore`, so they won't be committed to the repository.

## Overview

### Hardware Requirements

- VRAM: 12GB or more recommended for image training, 24GB or more for video training
    - *Actual requirements depend on resolution and training settings.* For 12GB, use a resolution of 960x544 or lower and use memory-saving options such as `--blocks_to_swap`, `--fp8_llm`, etc.
- Main Memory: 64GB or more recommended, 32GB + swap may work

### Features

- Memory-efficient implementation
- Windows compatibility confirmed (Linux compatibility confirmed by community)
- Multi-GPU training (using [Accelerate](https://huggingface.co/docs/accelerate/index)), documentation will be added later

### Documentation

For detailed information on specific architectures, configurations, and advanced features, please refer to the documentation below.

**Architecture-specific:**

- [MiniMax H3](./docs/minimax_h3.md)
- [HunyuanVideo](./docs/hunyuan_video.md)
- [Wan2.1/2.2](./docs/wan.md)
- [Wan2.1/2.2 (Single Frame)](./docs/wan_1f.md)
- [FramePack](./docs/framepack.md)
- [FramePack (Single Frame)](./docs/framepack_1f.md)
- [FLUX.1 Kontext](./docs/flux_kontext.md)
- [Qwen-Image](./docs/qwen_image.md)
- [Z-Image](./docs/zimage.md)
- [HiDream-O1-Image](./docs/hidream_o1.md)
- [HunyuanVideo 1.5](./docs/hunyuan_video_1_5.md)
- [Kandinsky 5](./docs/kandinsky5.md)
- [FLUX.2](./docs/flux_2.md)

**Common Configuration & Usage:**
- [Dataset Configuration](./docs/dataset_config.md)
- [Advanced Configuration](./docs/advanced_config.md)
- [Sampling during Training](./docs/sampling_during_training.md)
- [Block Swap (CPU Offloading for Memory Saving)](./docs/block_swap.md)
- [Tools and Utilities](./docs/tools.md)
- [Using torch.compile](./docs/torch_compile.md)

## Installation

### pip based installation

Python 3.10 or later is required (verified with 3.10).

Create a virtual environment and install PyTorch and torchvision matching your CUDA version. 

PyTorch 2.5.1 or later is required (see [note](#PyTorch-version)).

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Install the required dependencies using the following command.

```bash
pip install -e .
```

Optionally, you can use FlashAttention and SageAttention (**for inference only**; see [SageAttention Installation](#sageattention-installation) for installation instructions).

Optional dependencies for additional features:
- `ascii-magic`: Used for dataset verification
- `matplotlib`: Used for timestep visualization
- `tensorboard`: Used for logging training progress
- `prompt-toolkit`: Used for interactive prompt editing in Wan2.1 and FramePack inference scripts. If installed, it will be automatically used in interactive mode. Especially useful in Linux environments for easier prompt editing.

```bash
pip install ascii-magic matplotlib tensorboard prompt-toolkit
```

### uv based installation (experimental)

You can also install using uv, but installation with uv is experimental. Feedback is welcome.

1. Install uv (if not already present on your OS).

#### Linux/MacOS

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Follow the instructions to add the uv path manually until you restart your session...

#### Windows

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Follow the instructions to add the uv path manually until you reboot your system... or just reboot your system at this point.

## Model Download

Model download procedures vary by architecture. Please refer to the architecture-specific documents in the [Documentation](#documentation) section for instructions.

## Usage


### Dataset Configuration

Please refer to [here](./docs/dataset_config.md).

### Pre-caching

Pre-caching procedures vary by architecture. Please refer to the architecture-specific documents in the [Documentation](#documentation) section for instructions.

### Configuration of Accelerate

Run `accelerate config` to configure Accelerate. Choose appropriate values for each question based on your environment (either input values directly or use arrow keys and enter to select; uppercase is default, so if the default value is fine, just press enter without inputting anything). For training with a single GPU, answer the questions as follows:

```txt
- In which compute environment are you running?: This machine
- Which type of machine are you using?: No distributed training
- Do you want to run your training on CPU only (even if a GPU / Apple Silicon / Ascend NPU device is available)?[yes/NO]: NO
- Do you wish to optimize your script with torch dynamo?[yes/NO]: NO
- Do you want to use DeepSpeed? [yes/NO]: NO
- What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]: all
- Would you like to enable numa efficiency? (Currently only supported on NVIDIA hardware). [yes/NO]: NO
- Do you wish to use mixed precision?: bf16
```

*Note*: In some cases, you may encounter the error `ValueError: fp16 mixed precision requires a GPU`. If this happens, answer "0" to the sixth question (`What GPU(s) (by id) should be used for training on this machine as a comma-separated list? [all]:`). This means that only the first GPU (id `0`) will be used.

### Training and Inference

Training and inference procedures vary significantly by architecture. Please refer to the architecture-specific documents in the [Documentation](#documentation) section and the various configuration documents for detailed instructions.

## Miscellaneous

### SageAttention Installation

sdbsd has provided a Windows-compatible SageAttention implementation and pre-built wheels here:  https://github.com/sdbds/SageAttention-for-windows. After installing triton, if your Python, PyTorch, and CUDA versions match, you can download and install the pre-built wheel from the [Releases](https://github.com/sdbds/SageAttention-for-windows/releases) page. Thanks to sdbsd for this contribution.

For reference, the build and installation instructions are as follows. You may need to update Microsoft Visual C++ Redistributable to the latest version.

1. Download and install triton 3.1.0 wheel matching your Python version from [here](https://github.com/woct0rdho/triton-windows/releases/tag/v3.1.0-windows.post5).

2. Install Microsoft Visual Studio 2022 or Build Tools for Visual Studio 2022, configured for C++ builds.

3. Clone the SageAttention repository in your preferred directory:
    ```shell
    git clone https://github.com/thu-ml/SageAttention.git
    ```

4. Open `x64 Native Tools Command Prompt for VS 2022` from the Start menu under Visual Studio 2022.

5. Activate your venv, navigate to the SageAttention folder, and run the following command. If you get a DISTUTILS not configured error, set `set DISTUTILS_USE_SDK=1` and try again:
    ```shell
    python setup.py install
    ```

This completes the SageAttention installation.

### PyTorch version

If you specify `torch` for `--attn_mode`, use PyTorch 2.5.1 or later (earlier versions may result in black videos).

If you use an earlier version, use xformers or SageAttention.

## Disclaimer

This repository is unofficial and not affiliated with the official repositories of the supported architectures. 

This repository is experimental and under active development. While we welcome community usage and feedback, please note:

- This is not intended for production use
- Features and APIs may change without notice
- Some functionalities are still experimental and may not work as expected
- Video training features are still under development

If you encounter any issues or bugs, please create an Issue in this repository with:
- A detailed description of the problem
- Steps to reproduce
- Your environment details (OS, GPU, VRAM, Python version, etc.)
- Any relevant error messages or logs

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for details.

## License

Code under the `hunyuan_model` directory is modified from [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) and follows their license.

Code under the `hunyuan_video_1_5` directory is modified from [HunyuanVideo 1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) and follows their license.

Code under the `wan` directory is modified from [Wan2.1](https://github.com/Wan-Video/Wan2.1). The license is under the Apache License 2.0.

Code under the `frame_pack` directory is modified from [FramePack](https://github.com/lllyasviel/FramePack). The license is under the Apache License 2.0.

Other code is under the Apache License 2.0. Some code is copied and modified from Diffusers.
