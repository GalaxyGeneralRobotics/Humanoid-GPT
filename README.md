<div align="center">

# 🤖 Humanoid-GPT

### [CVPR 2026] Scaling Data and Structure for Zero-Shot Motion Tracking

<p align="center">
  <a href="https://cvpr.thecvf.com/Conferences/2026"><img src="https://img.shields.io/badge/CVPR-2026-4b44ce.svg?style=flat-square" alt="CVPR 2026"></a>
  <a href="https://arxiv.org/abs/2606.03985"><img src="https://img.shields.io/badge/arXiv-2606.03985-b31b1b.svg?style=flat-square" alt="arXiv"></a>
  <a href="https://qizekun.github.io/Humanoid-GPT/"><img src="https://img.shields.io/badge/Project-Page-blue.svg?style=flat-square" alt="Project Page"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?style=flat-square" alt="License"></a>
</p>

<p align="center">
  <img src="storage/assets/teaser.png" width="100%" alt="Humanoid-GPT Teaser">
</p>

</div>

---

## 📖 Overview

**Humanoid-GPT** is the first **GPT-style humanoid motion Transformer** trained with causal attention on a billion-scale motion corpus for whole-body control. Unlike prior shallow MLP trackers constrained by scarce data and an agility–generalization trade-off, Humanoid-GPT is pre-trained on a **2B-frame retargeted corpus** that unifies all major mocap datasets with large-scale in-house recordings.

<details>
<summary><b>🔬 Key Contributions</b></summary>

- **Billion-Scale Pre-Training**: First to scale humanoid motion learning to 2B frames
- **GPT-Style Architecture**: Causal Transformer with Rotary Position Embeddings (RoPE)
- **Zero-Shot Generalization**: Track arbitrary unseen motions without fine-tuning

</details>

### ✨ Highlights

| Feature             | Description                                                               |
|---------------------|---------------------------------------------------------------------------|
| 🧠 **Architecture** | Causal Transformer with RoPE, supporting variable-length motion sequences |
| 📊 **Scale**        | Pre-trained on 2B motion frames from unified mocap datasets               |
| 🎯 **Zero-Shot**    | Unprecedented generalization to unseen motions and tasks                  |
| 🤖 **Platform**     | Optimized for Unitree G1 humanoid robot (29 DOF whole-body)               |
| ⚡  **Speed**        | GPU-accelerated simulation with MuJoCo-MJX                                |

---

## TODO

The following features are planned for future implementation:

- ✅ Inference & deployment code.
- ✅ Pre-trained model checkpoints (storage/ckpts/).
- Training code.
- Training data.

---

## 📦 Installation

### Prerequisites

- NVIDIA GPU with CUDA 12.x
- **MacOS** is also supported for testing if you skip **jax[cuda12]** and use **mjpython** (e.g. `mjpython -m scripts.app`).
- Conda / Miniconda

### Quick Start

```bash
git clone https://github.com/GalaxyGeneralRobotics/Humanoid-GPT.git
cd Humanoid-GPT

conda create -n h-gpt python=3.12 -y
conda activate h-gpt

pip install -e ".[cuda]"     # or ".[cpu]" on MacOS, or "." for real robot deploy-only
```

On MacOS, use `mjpython` instead of `python` for the MuJoCo viewer (e.g. `mjpython -m scripts.app`).

### 🔧 G1 Hardware Version

We support multiple Unitree G1 hardware versions via the `G1_VERSION` env var (default `5010`). The asset folder `storage/assets/unitree_g1_${G1_VERSION}/` is selected automatically:

```bash
G1_VERSION=5010 python -m scripts.inference ...                   # default: 5010
```

---

## 🚀 Inference & Evaluation

A pre-trained tracking policy (`.onnx`) and a sample trajectory under
`storage/test/` are all you need to get started.

```bash
# Interactive Gradio demo
python -m scripts.app

# Track a single motion / a folder of motions
python -m scripts.inference --onnx_track storage/ckpts/pns_wo_priv216.onnx --track_dir storage/test

# Parallel evaluation over a folder of trajectories
python -m scripts.eval_parallel --onnx_track storage/ckpts/pns_wo_priv216.onnx \
    --track_dir storage/test --workers 32 --privileged

# Visualize a reference trajectory
python -m scripts.vis --path storage/test
```

The expected motion format is a `.npz` containing either `qpos` directly, or
`root_pos` / `root_rot` / `dof_pos` arrays. To convert retargeted mocap into
the keypoint representation the policy consumes:

```bash
python tracking/convert_qpos2kpt.py --mocap_npz <mocap_path.npz> --debug   # single file (debug viz)
python tracking/convert_parallel.py --src_dir <in_dir> --save_dir <out_dir> --num_workers 32
```

---

## 🤖 Real-Robot Deployment

Deployment on Unitree G1 is split into sub-modules under `deploy/` — start with
**[`deploy/DEPLOY.md`](deploy/DEPLOY.md)** for install / SDK setup, then:

```bash
# Simulation
python -m deploy.play_track --track-dir storage/test

# Real robot
python -m deploy.play_track --real --net <nic_name>
```

- 🖥️ [`onboard_deploy/`](deploy/onboard_deploy/DEPLOY_ONBOARD.md) — on-board (Jetson Orin) deploy.
- 🖥️ `onboard_deploy_wo_GMR/` — on-board variant that streams retargeting from a host.
- ✋ [`brainco/`](deploy/brainco/BRAINCO.md) — BrainCo dexterous-hand tracking variant.

---

## 📁 Project Structure

```
Humanoid-GPT/
├── 📂 tracking/   # Inference core: constants, infer_utils, ONNX policy wrapper (policy.py),
│                  # keypoint conversion (convert_qpos2kpt.py) and tracking metrics
├── 📂 scripts/    # inference.py · eval_parallel.py · vis.py · app.py (gradio demo)
├── 📂 deploy/     # Real-robot deployment — see deploy/DEPLOY.md
│   ├── onboard_deploy/        # On-board (Jetson) SSH deployment
│   ├── onboard_deploy_wo_GMR/ # On-board variant with host-side retargeting
│   └── brainco/               # BrainCo dexterous-hand tracking variant
├── 📂 projects/   # Optional side modules
│   ├── hme/                  # Harmonic Motion Encoder (Periodic Autoencoder)
│   ├── gqs/                  # General Quality Selection (physics + diversity scoring)
│   └── transformer/          # Transformer tracking policy (inference / deploy)
├── 📂 utils/      # MuJoCo / MJX simulation, transforms, video rendering
└── 📂 storage/    # Assets, configs, sample trajectory, released checkpoints
```

---

## 📚 Citation

```bibtex
@article{humanoid-gpt26,
    title     = {Humanoid-GPT: Scaling Data and Structure for Zero-Shot Motion Tracking},
    author    = {Qi, Zekun and Chen, Xuchuan and Liu, Dairu and Lin, Chenghuai and Lian, Yunrui and
                 Liang, Sikai and Zhang, Zhikai and Guan, Yu and Wang, Jilong and Zhang, Wenyao and
                 Yu, Xinqiang and Wang, He and Yi, Li},
    journal   = {arXiv preprint arXiv:2606.03985},
    year      = {2026}
}
```

---

## 📄 License · Acknowledgments

Licensed under **Apache 2.0**. Built on top of [MuJoCo](https://mujoco.org/), [Brax](https://github.com/google/brax) and the [Unitree](https://www.unitree.com/) G1 platform.
