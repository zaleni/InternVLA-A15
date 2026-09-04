# InternVLA-A1.5: Unifying Understanding, Latent Foresight, and Action for Compositional Generalization

![InternVLA-A1.5 teaser](assets/teaser.png)

[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/pdf/2607.04988)
[![Project Page](https://img.shields.io/badge/Website-Pages-blue.svg)](https://internrobotics.github.io/internvla-a15.github.io)
[![HuggingFace](https://img.shields.io/badge/Data-HuggingFace-blue?logo=huggingface)](https://huggingface.co/collections/InternRobotics/internvla-a15)
[![ModelScope](https://img.shields.io/badge/ModelScope-Model-blue?logo=modelscope)](https://www.modelscope.cn/collections/InternRobotics/InternVLA-A15)
[![License](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](LICENSE)

> [!NOTE]
> The InternVLA-A1 code has moved to the `InternVLA-A1` branch.

## Highlights

> **InternVLA-A1.5** unifies vision-language **understanding**, visual **foresight**, and **action** generation in one policy.

![InternVLA-A1.5 model](assets/model.png)

- *The Core:* Attaches a lightweight unified action expert to a native Qwen3.5-2B VLM backbone through shared full-attention layers, while preserving modality-specific Gated DeltaNet processing.
- *The Foresight:* Uses learnable foresight tokens to query task-relevant future dynamics from the shared multimodal context, supervised by a frozen WAN2.2-5B video generation model during training.
- *The Output:* Discards the video branch at inference and predicts continuous action chunks through flow matching, keeping deployment latency practical.

> **InternVLA-A1.5** delivers strong performance across broad simulation benchmarks and real-world manipulation settings.

| RoboTwin 2.0 | LIBERO | LIBERO-Plus | SimplerEnv | DOMINO | EBench |
|--------------|--------|-------------|------------|--------|--------|
| 93.2 | 98.9 | 84.8 | 80.8| 27.7 | 35.2|

| Model | HuggingFace | ModelScope |
|-------|-------------|------------|
| InternVLA-A1.5-base | [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?logo=huggingface)](https://huggingface.co/InternRobotics/InternVLA-A1.5-base) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-blue?logo=modelscope)](https://www.modelscope.cn/models/InternRobotics/InternVLA-A1.5-base) |
| InternVLA-A1.5-RoboTwin | [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?logo=huggingface)](https://huggingface.co/InternRobotics/InternVLA-A1.5-RoboTwin) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-blue?logo=modelscope)](https://www.modelscope.cn/models/InternRobotics/InternVLA-A1.5-RoboTwin) |
| InternVLA-A1.5-Libero | [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?logo=huggingface)](https://huggingface.co/InternRobotics/InternVLA-A1.5-Libero) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-blue?logo=modelscope)](https://www.modelscope.cn/models/InternRobotics/InternVLA-A1.5-Libero) |
| InternVLA-A1.5-DOMINO | [![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?logo=huggingface)](https://huggingface.co/InternRobotics/InternVLA-A1.5-DOMINO) | [![ModelScope](https://img.shields.io/badge/ModelScope-Model-blue?logo=modelscope)](https://www.modelscope.cn/models/InternRobotics/InternVLA-A1.5-DOMINO) |

## TODO List

- [x] Release InternVLA-A1.5 training and evaluation code
- [x] Release pre-training and fine-tuning tutorials
- [x] Release simulation evaluation entries for RoboTwin, LIBERO, LIBERO-Plus, DOMINO, and SimplerEnv
- [ ] Combine InternVLA-A1, InternVLA-A1.5 and more VLAs and WAMs in the repo
- [ ] Release the example VQA data and training tutorials

## Table of Contents

- [Installation](#installation)
- [Playground](#playground)
- [Pre-training](#pre-training)
- [Fine-tuning](#fine-tuning)
  - [Fine-tuning on LeRobot V2.1 dataset](#fine-tuning-on-lerobot-v21-dataset)
  - [Fine-tuning on RoboTwin 2.0](#fine-tuning-on-robotwin-20)
- [Evaluation & Inference](#evaluation--inference)

---

## Installation

This repository has been tested on **Python 3.11**, **CUDA 12.8**, and **PyTorch 2.10.0**.
We recommend using **conda** to create an isolated environment.

Please refer to the [Installation Tutorial](tutorials/installation.md) to prepare your environment, install dependencies, and patch the required HuggingFace Transformers modules for Qwen3.5 and the robot-learning policies.

---

## Playground

### Quick start with `lerobot/pusht`

#### One-line command

```bash
bash launch/internvla_a15_finetune.sh lerobot/pusht abs false
```

Here, **`abs`** indicates using **absolute actions**, and **`false`** means that the training script will use the **statistics file (`stats.json`) provided by `lerobot/pusht` itself**.

---

## Pre-Training

Please refer to the [Pre-training Tutorial](tutorials/pretrain_internvla_a1_5_with_interndata_a1.md) for instructions on pretraining InternVLA-A1.5 with the [**InternData-A1**](https://huggingface.co/datasets/InternRobotics/InternData-A1) dataset.

The pre-training entry uses `launch/internvla_a15_pretrain.sh`, discovers LeRobot-format datasets under `data/a1`, applies `configs/weight_rules_pretrain.yaml`, and trains the `internvla_a1_5` policy with Qwen3.5-2B, FAST action tokens, and the WAN video auxiliary branch.

---

## Fine-tuning

### Fine-tuning on LeRobot V2.1 dataset

Please refer to the [LeRobot V2.1 Fine-tuning Tutorial](tutorials/finetune_on_lerobot_v21_dataset.md) to fine-tune InternVLA-A1.5 with real-world datasets in the LeRobot V2.1 format.

This guide walks through the complete pipeline: download dataset -> convert to LeRobot V3.0 format -> compute delta-action statistics -> fine-tune InternVLA-A1.5.

### Fine-tuning on RoboTwin 2.0

Benchmark InternVLA-A1.5 on RoboTwin 2.0 with the [RoboTwin 2.0 Fine-tuning Tutorial](tutorials/finetune_internvla_a1_5_with_robotwin.md) and the [RoboTwin 2.0 Eval Tutorial](evaluation/RoboTwin/README.md).

### Using an AHA-WAM Video Expert as the foresight teacher

The training-only WAN branch can load the Video Expert from a released AHA-WAM
checkpoint. AHA-WAM checkpoints store it under `mot/mixtures.video.*`; the
loader detects that format, drops the ActionDiT/editor weights, validates every
Video Expert parameter name and shape, and enables AHA-WAM's frame-causal
attention and clean-first-latent timestep semantics.

For distributed training, first extract a compact Video Expert checkpoint so
that every rank does not load the unused AHA-WAM action weights:

```bash
python util_scripts/extract_aha_wan_video_expert.py \
    --checkpoint /path/to/aha_wam_step.pt \
    --wan-config-dir "${HF_HOME}/hub/Wan2.2-TI2V-5B" \
    --output /path/to/aha_wan_video_expert.pt
```

The launchers keep the checkpoint, architecture config, and VAE paths separate.
For RoboTwin, the dedicated wrapper supplies the required AHA teacher settings
and follows the original RoboTwin fine-tuning configuration while unfreezing
the foresight/context adapter needed to learn AHA's conditioning distribution.
The AHA Video Expert itself stays frozen. By default it uses all 100 repos under
`/data/jjhao/data/robotwin`: clean_50 plus randomized_500 for 50 tasks, totaling
27,500 episodes.

The dataset transform can read each repo's `meta/annotations.jsonl` and use the
subtask for the segment containing the current frame as language supervision.
For the paired RoboTwin Wan2.2/AHA experiments below this is deliberately
disabled (`USE_SUBTASK_ANNOTATIONS=false`) so subtask supervision is not an
additional experimental variable.

```bash
bash launch/internvla_a15_finetune_robotwin_aha_teacher.sh
```

This uses
`/data/jjhao/data/model/AHA-WAM-RoboTwin2.0/robotwin_ahawam_video_expert.pt`
by default; a positional path still overrides it.

The matching original-Wan2.2 baseline is:

```bash
bash launch/internvla_a15_finetune_robotwin_wan22_baseline.sh
```

Both wrappers default to the same local A1.5 pretrain checkpoint, all 27,500
RoboTwin episodes, 2 nodes x 8 GPUs, batch size 8 per GPU, and 60,000 optimizer
steps. They also share LR `5e-5`, 2,000 warmup steps, cosine decay to `5e-6`,
seed 42, action/video loss weights, normalization statistics, and checkpoint
frequency. The intended teacher-specific differences are that the baseline
uses standard Wan2.2 semantics and freezes the original foresight tokens,
whereas the AHA run uses AHA attention/timestep semantics and trains the
foresight/context adapter; both video DiTs stay frozen.

As with the AC-One launcher, submit one command to a 2-node x 8-GPU platform
job. The platform runs it on both nodes and injects `MASTER_ADDR`,
`MASTER_PORT`, `SENSECORE_PYTORCH_NNODES`, and
`SENSECORE_PYTORCH_NODE_RANK`. The baseline task command is:

```bash
bash /data/jjhao/InternVLA-A-series/launch/internvla_a15_finetune_robotwin_wan22_baseline.sh
```

The AHA task command is:

```bash
bash /data/jjhao/InternVLA-A-series/launch/internvla_a15_finetune_robotwin_aha_teacher.sh
```

Do not submit separate node-0/node-1 commands; node rank and rendezvous values
come from the platform job environment.

To smoke-test a shorter run while still sampling all 27,500 episodes:

```bash
STEPS=100 \
WANDB_ENABLE=false \
bash launch/internvla_a15_finetune_robotwin_aha_teacher.sh \
    /path/to/another_aha_wan_video_expert.pt
```

The default dataset root is `/data/jjhao/data`; override `DATASET_ROOT`,
`WAN_CONFIG_PATH`, `WAN_VAE_PATH`, or the other launcher environment variables
when your local layout differs. Set `DRY_RUN=true` to validate and print the
resolved command without starting training.

A raw AHA-WAM `.pt` can also be passed directly as `WAN_CHECKPOINT_PATH`, but
the extracted checkpoint is more memory- and I/O-efficient. This integration
replaces only the frozen video auxiliary teacher: normal action inference still
uses InternVLA's Action Expert and does not run WAN. The policy checkpoint does
not embed the frozen teacher, so the external checkpoint, config, and VAE must
remain available when resuming video-supervised training.

For AC-One, the dedicated wrapper sets all required teacher flags:

```bash
bash launch/internvla_a15_finetune_acone_aha_teacher.sh \
    /path/to/aha_wan_video_expert.pt
```

---

## Evaluation & Inference

Please refer to the evaluation guides for the complete inference and benchmark workflow:

- [RoboTwin Evaluation](evaluation/RoboTwin/README.md)
- [LIBERO Evaluation](evaluation/LIBERO/README.md)
- [LIBERO-Plus Evaluation](evaluation/LIBERO-plus/README.md)
- [DOMINO Evaluation](evaluation/DOMINO/README.md)
- [SimplerEnv Evaluation](evaluation/SimplerEnv/eval_simplerenv/README.md)

For open-loop action prediction on LeRobot-format data, see [`tests/openloop_internvla_a1_5.py`](tests/openloop_internvla_a1_5.py).

### Real-robot inference

For real-robot deployment, we recommend using the optimized action-only inference backend. After loading the InternVLA-A1.5 config and before constructing the policy, add:

```python
config.inference_backend, config.action_loss_only = "optimized", True
```

This skips WAN video-branch loading and uses the low-latency optimized backend for action prediction. Keep the standard backend only when you need future-video visualization or WAN-related inference outputs.

## License and Citation

All code within this repo is released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Please consider citing our project if it helps your research.

```BibTeX
@article{internvla_a15,
  title={InternVLA-A1.5: Unifying Understanding, Latent Foresight, and Action for Compositional Generalization},
  author={Ma, Haoxiang and Cai, Junhao and Xu, Xiaoxu and Li, Hao and Yang, Yuyin and Tian, Yang and Cao, Jiafei and Zhu, Hongrui and Qiu, Zherui and others},
  journal={arXiv preprint arXiv:2607.04988}
  year={2026},
}

@article{internvla_a1,
  title={InternVLA-A1: Unifying Understanding, Generation and Action for Robotic Manipulation},
  author={Cai, Junhao and Cai, Zetao and Cao, Jiafei and Chen, Yilun and He, Zeyu and Jiang, Lei and Li, Hang and Li, Hengjie and Li, Yang and Liu, Yufei and others},
  journal={arXiv preprint arXiv:2601.02456},
  year={2026}
}
```

## Contact

If you have any questions, feel free to submit GitHub issues or email mahaoxiang@pjlab.org.cn.

## Acknowledgments

- [LeRobot](https://github.com/huggingface/lerobot)
- [openpi](https://github.com/Physical-Intelligence/openpi)
- [InternVLA-A1](https://github.com/InternRobotics/InternVLA-A1)
- [Qwen3.5](https://github.com/QwenLM/Qwen3.6)
- [Wan](https://github.com/Wan-Video/Wan2.2)
