"""Self-contained smoke test for the in-repo gmr package.

Two things are checked. First the retarget model invariants: `g1_mocap_track.xml`
must carry the massless toe IK-target markers, the 0.002 s IK integration
timestep, and joint ranges that are a strict subset of the tracking model's
(`g1_mjx_track.xml`) -- that subset relation is what makes every retarget
reachable by the trained policy. Then that all three sources retarget to finite
qpos which respects those narrowed limits.

No external repos required. Run from repo root with the h-gpt env:

    python -m gmr.tests.test_smoke
"""
import sys
import pathlib

import numpy as np
import mujoco as mj

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import gmr
from gmr import GeneralMotionRetargeting as GMR, ROBOT_XML_DICT
from gmr.sources import load_lafan1_file

REPO = pathlib.Path(__file__).resolve().parents[2]
BVH = REPO / "mocap.bvh"

IK_TIMESTEP = 0.002
TOE_MARKERS = ("left_toe_link", "right_toe_link")
# mink integrates the QP solution, so a solve that saturates a limit can land a
# few ulps outside it.
LIMIT_TOL = 1e-6

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


def _hinge_ranges(model):
    """{joint_name: (lo, hi)} for every non-free joint, in model order."""
    out = {}
    for i in range(model.njnt):
        if model.jnt_type[i] == mj.mjtJoint.mjJNT_FREE:
            continue
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
        out[name] = (float(model.jnt_range[i][0]), float(model.jnt_range[i][1]))
    return out


def check_model(ik_xml):
    """Verify the IK model against the tracking model it must stay reachable for."""
    track_xml = ik_xml.with_name("g1_mjx_track.xml")
    assert track_xml.exists(), f"tracking model missing: {track_xml}"

    ik = mj.MjModel.from_xml_path(str(ik_xml))
    track = mj.MjModel.from_xml_path(str(track_xml))

    bodies = {mj.mj_id2name(ik, mj.mjtObj.mjOBJ_BODY, i) for i in range(ik.nbody)}
    missing = [b for b in TOE_MARKERS if b not in bodies]
    ok = not missing
    print(f"[model] toe markers present={not missing}"
          + (f" missing={missing}" if missing else ""))

    ts_ok = abs(ik.opt.timestep - IK_TIMESTEP) < 1e-12
    print(f"[model] IK timestep={ik.opt.timestep} (expected {IK_TIMESTEP}) ok={ts_ok}")
    ok &= ts_ok

    nq_ok = ik.nq == 36
    print(f"[model] nq={ik.nq} ok={nq_ok}")
    ok &= nq_ok

    ik_r, track_r = _hinge_ranges(ik), _hinge_ranges(track)
    joints_ok = list(ik_r) == list(track_r)
    print(f"[model] joint set/order matches tracking model={joints_ok}")
    ok &= joints_ok

    outside = [
        (j, ik_r[j], track_r[j])
        for j in ik_r
        if j in track_r
        and (ik_r[j][0] < track_r[j][0] - LIMIT_TOL
             or ik_r[j][1] > track_r[j][1] + LIMIT_TOL)
    ]
    narrower = sum(1 for j in ik_r if j in track_r and ik_r[j] != track_r[j])
    print(f"[model] IK ranges within tracking ranges={not outside}, "
          f"narrowed on {narrower}/{len(ik_r)} joints")
    for j, a, b in outside:
        print(f"          WIDER THAN TRACKING: {j} ik={a} track={b}")
    ok &= not outside and narrower > 0
    return ok, ik_r


def _in_limits(qpos, ik_ranges):
    """Largest amount by which the dof block of `qpos` leaves the IK limits."""
    lo = np.array([r[0] for r in ik_ranges.values()])
    hi = np.array([r[1] for r in ik_ranges.values()])
    dof = qpos[:, 7:]
    return float(np.maximum(lo - dof, dof - hi).max())


def main():
    print("gmr:", gmr.__file__)
    model = ROBOT_XML_DICT["unitree_g1"]
    assert model.exists(), f"model missing: {model}"
    print("model:", model)

    ok, ik_ranges = check_model(model)

    # bvh from the real mocap.bvh (untracked local capture, so it may be absent)
    if BVH.exists():
        frames, h = load_lafan1_file(str(BVH), pns=True)
        r = GMR(src_human="bvh", tgt_robot="unitree_g1", actual_human_height=h)
        q = np.stack([r.retarget(f) for f in frames[3:23]])
        fin = np.isfinite(q).all()
        viol = _in_limits(q, ik_ranges)
        print(f"[bvh]        qpos={q.shape} finite={fin} max_limit_violation={viol:.2e}")
        ok &= fin and q.shape[1] == 36 and viol <= LIMIT_TOL
    else:
        print(f"[bvh]        SKIP (no {BVH.name} in repo root)")

    # noitom / xsens from synthetic frames
    for src in ("fbx_noitom", "fbx_xsens"):
        r = GMR(src_human=src, tgt_robot="unitree_g1", actual_human_height=1.7)
        q = np.stack([r.retarget(f) for f in _synth()])
        fin = np.isfinite(q).all()
        viol = _in_limits(q, ik_ranges)
        print(f"[{src:10s}] qpos={q.shape} finite={fin} max_limit_violation={viol:.2e}")
        ok &= fin and q.shape[1] == 36 and viol <= LIMIT_TOL

    print("SMOKE OK" if ok else "SMOKE FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
