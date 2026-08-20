# GMR for Humanoid-GPT

Frame-wise General Motion Retargeting (GMR) for Unitree G1. It supports
offline OptiTrack/LAFAN and Noitom PNS BVH files, plus the Noitom and Xsens
real-time sources used by `deploy/`.

## Setup

Install the repository and GMR's IK dependency:

```bash
pip install -e .
pip install mink==1.1.0
```

GMR solves IK on `storage/assets/unitree_g1_<version>/g1_mocap_track.xml`, which
is `g1_mjx_track.xml` plus the massless toe IK-target markers. Its joint ranges
are narrowed to the original GMR-galbot limits, a strict subset of the ranges the
tracking model `g1_mjx_track.xml` uses, so a retarget always stays reachable for
the trained policy while the policy itself keeps the wider range to work in.
`G1_VERSION` selects the asset directory (default: `5010`), and `GMR_ASSET_ROOT`
can point to a different asset root.

## Batch BVH conversion

```bash
python -m gmr.scripts.bvh_to_robot_parallel \
    --bvh-dir bvhs/ \
    --out-dir gmr_out/ \
    --num_proc 8 \
    --assume-format auto
```

The command recursively preserves the input directory layout. `auto` detects
Noitom PNS versus OptiTrack/LAFAN; use `pns` or `optitrack` to force it.
Existing outputs are skipped unless `--overwrite` is passed. The default is
29 DoF with the original Savitzky-Golay smoothing enabled; use `--save_23dof`
for 23 DoF or `--no-smooth` to disable smoothing.

Inputs are deliberately filtered to 90--120 Hz, 51- or 59-joint BVHs, and a
non-degenerate parsed skeleton. Each result contains `qpos` with shape
`[T, 36]` (`root_pos[3]`, scalar-first `root_quat[4]`, 29 joints), its source
`frequency`, zero `qvel`, `split_points`, and retarget-model joint metadata.
Use `tracking/convert_parallel.py` to convert these archives to the tracking
keypoint format and target frequency.

## Python API

```python
from gmr import GeneralMotionRetargeting
from gmr.sources import load_lafan1_file

frames, human_height = load_lafan1_file("motion.bvh", pns=False)
retargeter = GeneralMotionRetargeting(
    src_human="bvh", tgt_robot="unitree_g1", actual_human_height=human_height
)
qpos = retargeter.retarget(frames[0])
```

A frame maps source body names to `(position[3], orientation_wxyz[4])`.
Supported `src_human` values are `bvh`, `fbx_noitom`, and `fbx_xsens`; their
source-specific task and offset definitions are in `ik_configs/`. Real-time
deployment wires the latter two sources through `deploy/`.

GMR solves each frame from the preceding robot configuration. It therefore
keeps temporal continuity but does not perform a trajectory-wide optimization.
