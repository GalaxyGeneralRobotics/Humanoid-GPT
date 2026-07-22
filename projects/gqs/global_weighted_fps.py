import os
import tyro
import torch
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from projects.hme.dataset import compute_win_len
from projects.hme.model import PeriodicAutoencoder


@dataclass
class Args:
    # Input/output paths
    mocap_dir: str = "storage/mocap/amass_train_pass"
    output_base_dir: str = "storage/mocap/amass_pass_global/"
    hme_ckpt: str = "storage/hme_ckpt/amass_pass.pt"
    # PAE model hyperparameters
    state_dim: int = 74       # qpos(36) + qvel(35) + gv_vel(3)
    phase_dim: int = 8
    win_sec: float = 4.0      # window duration in seconds
    downsample_rate: int = 5
    device: str = "cuda"
    # Sampling parameters
    # selection_ratio is dataset-dependent (paper: ~10% optimal for AMASS,
    # ~30% for PHUMA via Adaptive Ratio Selection r* = 0.5 * EDR^2).
    selection_ratio: float = 0.1
    # alpha weights diversity (1.0) vs complexity (0.0); paper Table 5 uses 0.99,
    # i.e. complexity acts only as a small tie-break bias.
    alpha: float = 0.99


def extract_windows(npz_path, win_sec, downsample_rate, state_dim, device):
    """
    Extract sliding windows from motion file.
    Returns: Tensor of shape (num_windows, win_len, state_dim)
    """
    data = np.load(npz_path)
    frequency = float(data["frequency"]) if "frequency" in data else 50.0
    
    qpos = torch.from_numpy(data["qpos"].astype(np.float32))
    qvel = torch.from_numpy(data["qvel"].astype(np.float32))
    gv_vel = torch.from_numpy(data["gv_vel"].astype(np.float32))
    features = torch.cat([qpos, qvel, gv_vel], dim=-1)  # (T, 74)
    
    win_len = compute_win_len(win_sec, downsample_rate, frequency)
    win_len_half = (win_len - 1) // 2
    padding = win_len_half * downsample_rate
    trj_len = features.shape[0]
    seq_len = trj_len - 2 * padding
    
    if seq_len <= 0:
        return None
    
    win_offset = torch.arange(-win_len_half, win_len_half + 1) * downsample_rate
    ref_ids = padding + torch.arange(seq_len)
    win_ids = ref_ids.unsqueeze(1) + win_offset
    
    windows = features[win_ids]  # (seq_len, win_len, 74)
    return windows.to(device)


def compute_complexity(file_path: str):
    """
    Compute physical complexity of motion (Energy / Dynamics Entropy).
    Metric = Mean(Vel^2) + 0.05 * Mean(Acc^2)
    """
    try:
        data = np.load(file_path)
        qvel = data['qvel']  # [T, D]

        # Get sampling frequency (default 50Hz)
        freq = float(data['frequency']) if 'frequency' in data else 50.0
        dt = 1.0 / freq

        T = qvel.shape[0]
        if T < 5:
            return 0.0

        # Compute acceleration
        qacc = np.diff(qvel, axis=0) / dt

        # Compute energy (Vel Power + Acc Power)
        vel_power = np.mean(np.sum(qvel**2, axis=1))
        acc_power = np.mean(np.sum(qacc**2, axis=1))

        return vel_power + 0.05 * acc_power
    except Exception:
        return 0.0


def weighted_fps_global(embeddings, complexities, n_samples, alpha=0.99):
    """
    Perform global weighted Farthest Point Sampling (FPS).
    """
    N = len(embeddings)
    if n_samples >= N:
        return list(range(N))

    selected_indices = []
    # min_dists records the shortest distance from each point to the selected set
    # Initialize to infinity
    min_dists = np.full(N, np.inf)

    # First point: the point with highest global complexity (as global anchor)
    first_idx = np.argmax(complexities)
    selected_indices.append(first_idx)

    # Initialize distance (distance to the first point)
    dists = np.linalg.norm(embeddings - embeddings[first_idx], axis=1)
    min_dists = np.minimum(min_dists, dists)

    # Show sampling progress with tqdm
    for _ in tqdm(range(n_samples - 1), desc="Global FPS Sampling"):
        # Dynamic distance normalization
        max_dist = np.max(min_dists)
        if max_dist > 1e-6:
            norm_dists = min_dists / max_dist
        else:
            norm_dists = min_dists

        # Combined scoring
        scores = alpha * norm_dists + (1 - alpha) * complexities

        # Mask already selected
        scores[selected_indices] = -1.0

        # Greedy selection
        next_idx = np.argmax(scores)
        selected_indices.append(next_idx)

        # Update minimum distances
        new_dists = np.linalg.norm(embeddings - embeddings[next_idx], axis=1)
        min_dists = np.minimum(min_dists, new_dists)

    return selected_indices


def main(args: Args):
    print(f"=" * 60)
    print(f"Global Weighted FPS Selection Pipeline")
    print(f"=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ========== Step 1: Load HME Model ==========
    print(f"\n[Step 1] Loading HME checkpoint: {args.hme_ckpt}")
    ckpt = torch.load(args.hme_ckpt, map_location=device)

    win_len = compute_win_len(args.win_sec, args.downsample_rate)
    model = PeriodicAutoencoder(
        args.state_dim,
        args.phase_dim,
        win_len=win_len,
        win_sec=args.win_sec,
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()

    # ========== Step 2: Scan Data and Compute Embeddings & Complexity ==========
    print(f"\n[Step 2] Scanning {args.mocap_dir}...")
    file_paths = sorted(list(Path(args.mocap_dir).rglob("*.npz")))
    print(f"Found {len(file_paths)} files.")

    filenames = []
    embeddings = []
    complexity_scores = []

    print("Computing embeddings and complexity...")
    with torch.no_grad():
        for p in tqdm(file_paths, desc="Processing"):
            try:
                # Extract sliding windows
                windows = extract_windows(
                    str(p),
                    args.win_sec,
                    args.downsample_rate,
                    args.state_dim,
                    device
                )
                if windows is None:
                    continue

                # Convert to model input format: [batch, feat, win_len]
                batch_input = windows.permute(0, 2, 1)  # [seq_len, state_dim, win_len]

                # Model encoding
                encoded = model.encode(batch_input)
                amp = encoded["amp"]   # [seq_len, phase_dim, 1]
                freq = encoded["freq"]  # [seq_len, phase_dim, 1]

                # Concatenate amp and freq as embedding
                window_embeddings = torch.cat([
                    amp.squeeze(-1),   # [seq_len, phase_dim]
                    freq.squeeze(-1)   # [seq_len, phase_dim]
                ], dim=-1)  # [seq_len, phase_dim * 2]

                # Average all windows to get global embedding for this motion
                global_emb = window_embeddings.mean(dim=0).cpu().numpy()  # [phase_dim * 2]

                # Compute complexity
                complexity = compute_complexity(str(p))

                embeddings.append(global_emb)
                complexity_scores.append(complexity)
                filenames.append(str(p))

            except Exception as e:
                print(f"Error processing {p.name}: {e}")
                continue

    # Convert to numpy arrays
    filenames = np.array(filenames)
    embeddings = np.array(embeddings)
    complexity_raw = np.array(complexity_scores, dtype=np.float32)

    print(f"\nProcessed {len(filenames)} valid samples.")
    print(f"Embedding shape: {embeddings.shape}")

    # ========== Step 3: Global Complexity Normalization (Rank Norm) ==========
    print(f"\n[Step 3] Normalizing complexity (Global Rank Norm)...")
    ranks = np.argsort(np.argsort(complexity_raw))
    complexity_norm = ranks / (len(ranks) - 1) if len(ranks) > 1 else ranks

    # ========== Step 4: Perform Global Weighted FPS Sampling ==========
    n_select = int(len(filenames) * args.selection_ratio)
    n_select = max(1, n_select)

    print(f"\n[Step 4] Running Global Weighted FPS to select {n_select} samples...")
    selected_indices = weighted_fps_global(embeddings, complexity_norm, n_select, alpha=args.alpha)

    final_indices = np.array(selected_indices)
    selected_filenames = filenames[final_indices]

    print(f"\nSelection Complete: {len(final_indices)} / {len(filenames)}")

    # ========== Step 5: Copy Files ==========
    ratio_percent = int(args.selection_ratio * 100)
    target_dir_name = f"global_{ratio_percent}%"
    output_dir = os.path.join(args.output_base_dir, target_dir_name)

    print(f"\n[Step 5] Copying files to: {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    copy_count = 0
    fail_count = 0

    for fname in tqdm(selected_filenames, desc="Copying Files"):
        fname_str = str(fname)
        src_path = fname_str

        if not os.path.exists(src_path):
            src_path = os.path.join(args.mocap_dir, os.path.basename(fname_str))

        if os.path.exists(src_path):
            dst_path = os.path.join(output_dir, os.path.basename(src_path))
            try:
                shutil.copy2(src_path, dst_path)
                copy_count += 1
            except Exception:
                fail_count += 1
        else:
            fail_count += 1

    # ========== Step 6: Save Metadata ==========
    df_res = pd.DataFrame({
        "filename": [os.path.basename(str(f)) for f in selected_filenames],
        "complexity_raw": complexity_raw[final_indices]
    })
    csv_path = os.path.join(output_dir, "dataset_details.csv")
    df_res.to_csv(csv_path, index=False)

    print(f"\n" + "=" * 60)
    print(f"Pipeline Complete!")
    print(f"=" * 60)
    print(f"Output Directory: {output_dir}")
    print(f"Total Samples:    {len(filenames)}")
    print(f"Selected:         {len(final_indices)} ({ratio_percent}%)")
    print(f"Copied:           {copy_count}")
    print(f"Missing:          {fail_count}")
    print(f"Metadata:         {csv_path}")
    print(f"=" * 60)


if __name__ == "__main__":
    main(tyro.cli(Args))
