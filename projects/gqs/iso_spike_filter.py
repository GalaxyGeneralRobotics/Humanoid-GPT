"""
Isolated Root-Speed Spike Filtering.

Detects discontinuous root-position jumps (isolated spikes) in mocap
trajectories and copies / hardlinks clean clips into an output directory.

A frame-to-frame root speed `v = ||Δxyz|| * fps` is an isolated spike when:
  v > thr  AND  max(neighbor speeds in ±neigh) < ratio * v

Reject a trajectory if it has any such spike (iso_count >= 1).

Frequency is read from `npz['frequency']` when present, otherwise parsed from
the filename pattern `NHz` (e.g. `..._90Hz_29dof.npz`), falling back to
`default_fps`.

Example:
  python -m projects.gqs.iso_spike_filter \\
      --mocap-dir mocap/self_all \\
      --output-dir mocap/self_all_iso
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tyro
from tqdm import tqdm

HZ_RE = re.compile(r"(\d+)Hz", re.I)


@dataclass
class Args:
    # Input/output paths
    mocap_dir: str = "storage/mocap/amass_train_convert"
    output_dir: str = "storage/mocap/amass_train_iso"
    # Spike detection parameters
    thr: float = 5.0
    """Root-speed threshold in m/s."""
    neigh: int = 3
    """Neighbor half-window (frames) for isolation check."""
    ratio: float = 0.35
    """Neighbor max speed must be < ratio * peak to count as isolated."""
    default_fps: float = 50.0
    """Fallback fps when neither npz field nor filename provides it."""
    # IO / runtime
    workers: int = 64
    hardlink: bool = True
    """Prefer hardlink (same filesystem); fall back to copy on failure."""
    report_dir: str = ""
    """Optional directory for summary.json / rejected.tsv (default: <output_dir>_report)."""


def parse_fps(name: str, default_fps: float) -> float:
    m = HZ_RE.search(name)
    return float(m.group(1)) if m else float(default_fps)


def has_isolated_spike(
    qpos: np.ndarray,
    fps: float,
    thr: float,
    neigh: int,
    ratio: float,
) -> Tuple[bool, int, float]:
    """Return (is_bad, iso_count, root_speed_max)."""
    if len(qpos) < 2:
        return False, 0, 0.0
    droot = np.linalg.norm(np.diff(qpos[:, :3], axis=0), axis=1) * float(fps)
    peak = float(droot.max()) if len(droot) else 0.0
    idx = np.where(droot > thr)[0]
    iso = 0
    for i in idx:
        lo = max(0, i - neigh)
        hi = min(len(droot), i + neigh + 1)
        neigh_vals = np.concatenate([droot[lo:i], droot[i + 1 : hi]])
        if len(neigh_vals) and neigh_vals.max() < ratio * droot[i]:
            iso += 1
    return iso >= 1, iso, peak


def _process_one(payload: Tuple) -> Tuple[str, str, str, int, float]:
    """Worker: (rel, src_root, dst_root, thr, neigh, ratio, default_fps, hardlink)."""
    rel_str, src_root, dst_root, thr, neigh, ratio, default_fps, hardlink = payload
    rel = Path(rel_str)
    src = Path(src_root) / rel
    try:
        data = np.load(src, allow_pickle=True)
        if "qpos" not in data.files:
            return ("error", rel_str, "no_qpos", 0, 0.0)
        q = np.asarray(data["qpos"], dtype=np.float64)
        if q.ndim != 2 or q.shape[1] < 3:
            return ("error", rel_str, f"bad_qpos_shape={q.shape}", 0, 0.0)
        if "frequency" in data.files:
            fps = float(np.asarray(data["frequency"]).reshape(-1)[0])
        else:
            fps = parse_fps(rel.name, default_fps)

        bad, iso, peak = has_isolated_spike(q, fps, thr, neigh, ratio)
        if bad:
            return ("reject", rel_str, f"iso={iso}", iso, peak)

        dst = Path(dst_root) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            try:
                if dst.samefile(src):
                    return ("keep_exists", rel_str, "already_linked", iso, peak)
            except Exception:
                pass
            if dst.stat().st_size == src.stat().st_size:
                return ("keep_exists", rel_str, "already_copied", iso, peak)
            dst.unlink()

        if hardlink:
            try:
                os.link(src, dst)
                return ("keep", rel_str, "hardlink", iso, peak)
            except OSError:
                pass
        shutil.copy2(src, dst)
        return ("keep_copy", rel_str, "copied", iso, peak)
    except Exception as e:
        return ("error", rel_str, f"{type(e).__name__}: {e}", 0, 0.0)


def run_filtering(args: Args) -> Dict:
    src_root = Path(args.mocap_dir)
    dst_root = Path(args.output_dir)
    if not src_root.exists():
        raise FileNotFoundError(f"Input directory not found: {args.mocap_dir}")

    print("=" * 60)
    print("Isolated Spike Filtering")
    print("=" * 60)
    print(f"Input Directory:   {args.mocap_dir}")
    print(f"Output Directory:  {args.output_dir}")
    print(f"thr={args.thr} m/s  neigh=±{args.neigh}  ratio={args.ratio}")
    print(f"workers={args.workers}  hardlink={args.hardlink}")
    print("=" * 60)

    print(f"Scanning source directory: {src_root}...")
    src_files = sorted(src_root.rglob("*.npz"))
    print(f"Found {len(src_files)} source files.")

    dst_root.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir) if args.report_dir else Path(str(dst_root) + "_report")
    report_dir.mkdir(parents=True, exist_ok=True)

    payloads = [
        (
            str(p.relative_to(src_root)),
            str(src_root),
            str(dst_root),
            args.thr,
            args.neigh,
            args.ratio,
            args.default_fps,
            args.hardlink,
        )
        for p in src_files
    ]

    counts = {"keep": 0, "keep_exists": 0, "keep_copy": 0, "reject": 0, "error": 0}
    rejected: List[Tuple[str, int, float, str]] = []
    errors: List[Tuple[str, str]] = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(_process_one, pl) for pl in payloads]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Filtering"):
            status, rel, msg, iso, peak = fut.result()
            counts[status] = counts.get(status, 0) + 1
            if status == "reject":
                rejected.append((rel, iso, peak, msg))
            elif status == "error":
                errors.append((rel, msg))

    kept = counts["keep"] + counts["keep_exists"] + counts["keep_copy"]
    summary = {
        "src": str(src_root),
        "dst": str(dst_root),
        "criterion": {
            "thr_mps": args.thr,
            "neigh": args.neigh,
            "ratio": args.ratio,
            "reject_if": "iso>=1",
            "fps_source": "npz['frequency'] or filename NHz or default_fps",
            "default_fps": args.default_fps,
        },
        "total": len(src_files),
        "kept": kept,
        "rejected": counts["reject"],
        "errors": counts["error"],
        "counts": counts,
        "elapsed_sec": time.time() - t0,
    }

    with open(report_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(report_dir / "rejected.tsv", "w") as f:
        f.write("relpath\tiso\troot_max\tmsg\n")
        for rel, iso, peak, msg in sorted(rejected):
            f.write(f"{rel}\t{iso}\t{peak:.4f}\t{msg}\n")
    with open(report_dir / "errors.tsv", "w") as f:
        f.write("relpath\tmsg\n")
        for rel, msg in sorted(errors):
            f.write(f"{rel}\t{msg}\n")

    print("-" * 50)
    print("Filtering Complete.")
    print(f"  Total Source:   {len(src_files)}")
    print(f"  Kept:           {kept}")
    print(f"    newly linked: {counts['keep']}")
    print(f"    newly copied: {counts['keep_copy']}")
    print(f"    already there:{counts['keep_exists']}")
    print(f"  Rejected:       {counts['reject']}")
    print(f"  Errors:         {counts['error']}")
    print(f"  Elapsed:        {summary['elapsed_sec']:.1f}s")
    print(f"Output Directory: {args.output_dir}")
    print(f"Report Directory: {report_dir}")
    print("-" * 50)
    return summary


def main(args: Args):
    run_filtering(args)
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main(tyro.cli(Args))
