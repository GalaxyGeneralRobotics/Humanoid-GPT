"""Generate the GMR retargeting model from a Humanoid-GPT robot version.

GMR retargets onto the *same* kinematics Humanoid-GPT trains/deploys on. The
deploy/sim model (`g1_mjx_track.xml`) lacks the massless IK-target markers the
noitom/xsens configs reference (`left/right_toe_link`). This script derives
`g1_mocap_track.xml` from it by:

  * adding `left/right_toe_link` as massless children of `*_ankle_roll_link`
    at the canonical local offset (0.1, 0, -0.02), matching the original
    GMR-galbot mocap model (identical toe world position);
  * setting `opt.timestep = 0.002` — for GMR this is purely the IK integration
    step (no physics is simulated); 0.002 matches the value the IK configs were
    tuned with.

The result shares the version's meshes (no duplicated assets) and has joints /
axes / body orientations identical to the deploy model.

Usage:
    python -m gmr.tools.build_mocap_model --version 5010
"""
import argparse
import pathlib

import mujoco as mj

TOE_LOCAL_POS = [0.1, 0.0, -0.02]
IK_TIMESTEP = 0.002

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # Humanoid-GPT/


def _get_body(spec, name):
    try:
        b = spec.body(name)
        if b is not None:
            return b
    except Exception:
        pass
    for b in spec.bodies:
        if b.name == name:
            return b
    raise KeyError(f"body {name!r} not found")


def build(version: str, asset_root: pathlib.Path | None = None) -> pathlib.Path:
    asset_root = asset_root or (REPO_ROOT / "storage" / "assets")
    ver_dir = asset_root / f"unitree_g1_{version}"
    src = ver_dir / "g1_mjx_track.xml"
    out = ver_dir / "g1_mocap_track.xml"
    if not src.exists():
        raise FileNotFoundError(src)

    spec = mj.MjSpec.from_file(str(src))
    for side in ("left", "right"):
        ankle = _get_body(spec, f"{side}_ankle_roll_link")
        toe = ankle.add_body()
        toe.name = f"{side}_toe_link"
        toe.pos = TOE_LOCAL_POS
    spec.option.timestep = IK_TIMESTEP
    spec.compile()
    out.write_text(spec.to_xml())

    # sanity reload
    m = mj.MjModel.from_xml_path(str(out))
    have = {mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)}
    assert {"left_toe_link", "right_toe_link"} <= have, "toe markers missing"
    assert m.nq == 36, f"unexpected nq={m.nq}"
    print(f"[build_mocap_model] wrote {out}  (nq={m.nq}, nbody={m.nbody}, "
          f"timestep={m.opt.timestep})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="5010", help="G1 robot version under storage/assets")
    args = ap.parse_args()
    build(args.version)
