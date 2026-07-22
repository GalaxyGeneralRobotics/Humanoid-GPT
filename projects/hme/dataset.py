import shutil
import zipfile
import concurrent.futures

import tqdm
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


# Features fed to the PAE: qpos(36) + qvel(35) + gv_vel(3) = 74.
_KEYS = ("qpos", "qvel", "gv_vel")
# Cache directory names to ignore when scanning the corpus.
_CACHE_DIRS = (".mmap_cache", ".hme_cache")


def compute_win_len(win_sec: float, downsample_rate: int, frequency: float = 50) -> int:
    """Compute window length from duration, frequency and downsample rate."""
    return int(win_sec * frequency / downsample_rate) + 1


def _find_npz(npz_path: str) -> list[Path]:
    """Sorted ``*.npz`` under ``npz_path`` (skips mmap/hme cache dirs).

    Same enumeration/order as ``tracking.train._find_npz_trajectories``, so a
    ``.mmap_cache`` built by the tracker aligns 1:1 with this scan.
    """
    return sorted(
        p
        for p in Path(npz_path).rglob("*.npz")
        if not any(c in p.parts for c in _CACHE_DIRS)
    )


def _read_npy_shape(zf: zipfile.ZipFile, name: str) -> tuple:
    """Read a .npy array shape from a zip entry without decompressing data."""
    with zf.open(name) as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, _, _ = np.lib.format.read_array_header_2_0(f)
        else:
            shape, _, _ = np.lib.format._read_array_header(f, version)
    return shape


class PAEDataset(Dataset):
    """Memory-mapped window dataset for PAE training.

    All trajectories are stored as flat, concatenated per-key ``.npy`` files that
    are memory-mapped, so ``__getitem__`` slices a window directly by absolute
    frame index — no per-sample ``np.load`` / decompression and no repeated reads
    of the same file (the previous design decompressed a whole trajectory for
    every single window).

    The concatenated buffer is resolved in this order:
      1. reuse the tracker's ``<root>/.mmap_cache`` (built by ``tracking.train``)
         if it is complete and its trajectory count matches this scan;
      2. reuse a private ``<root>/.hme_cache`` from a previous run;
      3. build ``<root>/.hme_cache`` from the npz files (one-time sequential pass).

    A single window length ``compute_win_len(win_sec, downsample_rate)`` is used
    for every window (matching the PAE's fixed Conv1d kernel), so all items batch
    together regardless of any per-file ``frequency`` metadata.
    """

    def __init__(
        self,
        npz_path: str,
        win_sec: float = 2.0,
        downsample_rate: int = 5,
        stride: int = 1,
        verbose: bool = True,
    ):
        self.win_sec = win_sec
        self.downsample_rate = downsample_rate

        npz_files = _find_npz(npz_path)
        assert npz_files, f"No .npz files found under {npz_path}."

        self._cache_dir, traj_lens = self._resolve_cache(
            Path(npz_path), npz_files, verbose
        )
        traj_start = np.concatenate([[0], np.cumsum(traj_lens)]).astype(np.int64)
        self._total_frames = int(traj_start[-1])

        # One global window length (matches the model's kernel), so every item
        # has identical shape and batches cleanly.
        win_len = compute_win_len(win_sec, downsample_rate)
        win_len_half = (win_len - 1) // 2
        padding = win_len_half * downsample_rate
        self._win_offset = (
            np.arange(-win_len_half, win_len_half + 1) * downsample_rate
        ).astype(np.int64)

        # Per (strided) trajectory: absolute frame of its first valid window
        # center, plus a global cumulative sample count for O(log n) indexing.
        centers = []
        self.cumsum = [0]
        for i in range(0, len(traj_lens), stride):
            num_samples = int(traj_lens[i]) - 2 * padding
            if num_samples > 0:
                centers.append(int(traj_start[i]) + padding)
                self.cumsum.append(self.cumsum[-1] + num_samples)
        self._centers = np.asarray(centers, dtype=np.int64)
        self.cumsum = np.asarray(self.cumsum, dtype=np.int64)
        assert len(self._centers) > 0, "No valid motion windows found."

        # Memmaps are opened lazily, once per (worker) process.
        self._mm: dict[str, np.memmap] | None = None

        if verbose:
            print(
                f"Found {len(self._centers)} trajectories (stride={stride}), "
                f"{int(self.cumsum[-1])} windows, {self._total_frames} frames "
                f"[mmap: {self._cache_dir}]"
            )

    # ------------------------------------------------------------------ cache
    def _resolve_cache(
        self, root: Path, npz_files: list[Path], verbose: bool
    ) -> tuple[Path, list[int]]:
        # 1) Reuse the tracker's mmap cache (meta.npz holds traj_lens).
        mm = root / ".mmap_cache"
        if (mm / "meta.npz").exists() and all(
            (mm / f"{k}.npy").exists() for k in _KEYS
        ):
            traj_lens = np.load(str(mm / "meta.npz"))["traj_lens"].tolist()
            if len(traj_lens) == len(npz_files):
                if verbose:
                    print(f"Reusing tracker mmap cache: {mm}")
                return mm, traj_lens
            if verbose:
                print(
                    f"tracker cache has {len(traj_lens)} trajs != {len(npz_files)} "
                    "found; building a private .hme_cache instead."
                )

        # 2) Reuse a private hme cache (traj_lens.npy, no .npz to avoid being
        #    picked up by the tracker's own *.npz scan).
        hc = root / ".hme_cache"
        if (hc / "traj_lens.npy").exists() and all(
            (hc / f"{k}.npy").exists() for k in _KEYS
        ):
            traj_lens = np.load(str(hc / "traj_lens.npy")).tolist()
            if len(traj_lens) == len(npz_files):
                if verbose:
                    print(f"Reusing hme mmap cache: {hc}")
                return hc, traj_lens

        # 3) Build the private cache.
        traj_lens = self._build_cache(hc, npz_files, verbose)
        return hc, traj_lens

    def _build_cache(
        self, cache_dir: Path, npz_files: list[Path], verbose: bool
    ) -> list[int]:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Pass 1: header-only scan for per-file frame counts + feature shapes.
        def _scan(p):
            with zipfile.ZipFile(str(p), "r") as zf:
                shapes = {k: _read_npy_shape(zf, f"{k}.npy") for k in _KEYS}
            return int(shapes["qpos"][0]), {k: shapes[k][1:] for k in _KEYS}

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
            it = pool.map(_scan, npz_files)
            if verbose:
                it = tqdm.tqdm(it, total=len(npz_files), desc="Scanning headers")
            scan = list(it)

        traj_lens = [r[0] for r in scan]
        key_shapes = scan[0][1]
        total = sum(traj_lens)

        # Pass 2: sequential write with read-ahead (fast even on network FS).
        fps = {}
        for k in _KEYS:
            fp = open(str(cache_dir / f"{k}.npy"), "wb", buffering=8 * 1024 * 1024)
            header = {
                "shape": (total, *tuple(key_shapes[k])),
                "fortran_order": False,
                "descr": np.lib.format.dtype_to_descr(np.dtype(np.float32)),
            }
            np.lib.format.write_array_header_2_0(fp, header)
            fps[k] = fp

        def _load(p):
            with np.load(str(p), allow_pickle=True) as z:
                return {k: z[k].astype(np.float32) for k in _KEYS}

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            it = pool.map(_load, npz_files)  # preserves order
            if verbose:
                it = tqdm.tqdm(it, total=len(npz_files), desc="Building mmap cache")
            for data in it:
                for k in _KEYS:
                    fps[k].write(data[k].tobytes())
        for fp in fps.values():
            fp.flush()
            fp.close()

        np.save(str(cache_dir / "traj_lens.npy"), np.asarray(traj_lens, dtype=np.int64))
        if verbose:
            print(f"Built hme mmap cache: {cache_dir} ({total} frames)")
        return traj_lens

    # --------------------------------------------------------------- indexing
    def _open(self) -> None:
        self._mm = {
            k: np.load(str(Path(self._cache_dir) / f"{k}.npy"), mmap_mode="r")
            for k in _KEYS
        }

    def __len__(self) -> int:
        return int(self.cumsum[-1])

    def __getitem__(self, idx):
        if self._mm is None:
            self._open()  # per-process lazy mmap open (safe with DataLoader workers)

        file_idx = int(np.searchsorted(self.cumsum[1:], idx, side="right"))
        local_idx = idx - int(self.cumsum[file_idx])
        ids = int(self._centers[file_idx]) + local_idx + self._win_offset  # (win_len,)

        feats = np.concatenate(
            [np.asarray(self._mm[k][ids]) for k in _KEYS], axis=-1
        )  # (win_len, 74)
        return torch.from_numpy(np.ascontiguousarray(feats, dtype=np.float32))
