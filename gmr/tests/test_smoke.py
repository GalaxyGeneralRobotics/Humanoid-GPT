"""Self-contained smoke test for the in-repo gmr package.

Confirms gmr loads the 5010 retarget model and retargets all three sources to
finite qpos. No external repos required. Run from repo root with the h-gpt env:

    python -m gmr.tests.test_smoke
"""
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import gmr
from gmr import GeneralMotionRetargeting as GMR, ROBOT_XML_DICT
from gmr.sources import load_lafan1_file

REPO = pathlib.Path(__file__).resolve().parents[2]
BVH = REPO / "mocap.bvh"

_BASE = {
    "Hips": (0, 0, 0.90), "Spine1": (0, 0, 1.10),
    "LeftUpLeg": (0.1, 0, 0.85), "LeftLeg": (0.1, 0, 0.45), "LeftFoot": (0.1, 0, 0.05),
    "RightUpLeg": (-0.1, 0, 0.85), "RightLeg": (-0.1, 0, 0.45), "RightFoot": (-0.1, 0, 0.05),
    "LeftArm": (0.2, 0, 1.3), "LeftForeArm": (0.35, 0, 1.2), "LeftHand": (0.5, 0, 1.1),
    "RightArm": (-0.2, 0, 1.3), "RightForeArm": (-0.35, 0, 1.2), "RightHand": (-0.5, 0, 1.1),
}


def _synth(n=20, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for t in range(n):
        fr = {}
        for b, p in _BASE.items():
            ang = 0.1 * np.sin(0.1 * t + len(b))
            q = np.array([np.cos(ang / 2), np.sin(ang / 2), 0, 0])
            fr[b] = (np.array(p, float) + 0.02 * np.sin(0.07 * t), q)
        frames.append(fr)
    return frames


def main():
    print("gmr:", gmr.__file__)
    model = ROBOT_XML_DICT["unitree_g1"]
    assert model.exists(), f"model missing: {model}"
    print("model:", model)

    ok = True
    # bvh from the real mocap.bvh
    frames, h = load_lafan1_file(str(BVH), pns=True)
    r = GMR(src_human="bvh", tgt_robot="unitree_g1", actual_human_height=h)
    q = np.stack([r.retarget(f) for f in frames[3:23]])
    fin = np.isfinite(q).all()
    print(f"[bvh]        qpos={q.shape} finite={fin}")
    ok &= fin and q.shape[1] == 36

    # noitom / xsens from synthetic frames
    for src in ("fbx_noitom", "fbx_xsens"):
        r = GMR(src_human=src, tgt_robot="unitree_g1", actual_human_height=1.7)
        q = np.stack([r.retarget(f) for f in _synth()])
        fin = np.isfinite(q).all()
        print(f"[{src:10s}] qpos={q.shape} finite={fin}")
        ok &= fin and q.shape[1] == 36

    print("SMOKE OK" if ok else "SMOKE FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
