import toml
import tyro
import tqdm
import torch
import numpy as np
from pathlib import Path
import torch.optim as optim
import matplotlib.pyplot as plt
from dataclasses import dataclass
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader

from projects.hme.model import PeriodicAutoencoder
from projects.hme.dataset import PAEDataset, compute_win_len

# Share tensors via the filesystem instead of file descriptors so many
# DataLoader workers don't exhaust the open-fd limit (avoids the pin_memory
# "Connection refused" crash on large corpora).
torch.multiprocessing.set_sharing_strategy("file_system")


@dataclass
class PAETrainConfig:
    mocap_dir: str = "storage/mocap/amass_train_convert"
    hme_ckpt: str = "storage/hme_ckpt/amass.pt"
    batch_size: int = 1024
    num_workers: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 5
    max_grad_norm: float = 10.0
    log_path: str = "storage/hme_log"
    device: str = "cuda:0"
    seed: int = 42
    stride: int = 1
    # PAE model hyperparameters
    state_dim: int = 74       # qpos(36) + qvel(35) + gv_vel(3)
    phase_dim: int = 8
    win_sec: float = 4.0      # window duration in seconds
    downsample_rate: int = 5

    def model_dump(self):
        return {
            "mocap_dir": self.mocap_dir,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "device": self.device,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "state_dim": self.state_dim,
            "phase_dim": self.phase_dim,
            "win_sec": self.win_sec,
            "downsample_rate": self.downsample_rate,
        }


def train_pae(config: PAETrainConfig) -> list:
    """Trains a model and returns the list of losses per epoch."""
    train_cfg_dict = config.model_dump()
    log_dir = Path(config.log_path) / Path(config.hme_ckpt).stem
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "pae_config.toml", "w", encoding="utf-8") as f:
        toml.dump({"train_config": train_cfg_dict}, f)
    print(f"Config saved to {log_dir}/pae_config.toml")

    # Dataset and DataLoader
    dataset = PAEDataset(
        config.mocap_dir,
        win_sec=config.win_sec,
        downsample_rate=config.downsample_rate,
        stride=config.stride,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
    )

    # Model
    win_len = compute_win_len(config.win_sec, config.downsample_rate)
    model = PeriodicAutoencoder(
        config.state_dim,
        config.phase_dim,
        win_len=win_len,
        win_sec=config.win_sec,
    )
    model.to(config.device)
    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    print(f"Starting training on {config.device} for {config.num_epochs} epochs...")
    loss_list = []
    for ep_id in range(1, config.num_epochs + 1):
        model.train()
        ep_loss_list = []
        for batch in tqdm.tqdm(dataloader, desc=f"Epoch {ep_id}"):
            # batch: (batch, win_len, feat) -> (batch, feat, win_len)
            inp = batch.permute(0, 2, 1).to(config.device)
            outputs = model(inp)
            loss = outputs["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            ep_loss_list.append(loss.item())

        avg_epoch_loss = np.mean(ep_loss_list)
        print(f"Epoch [{ep_id}/{config.num_epochs}], Loss: {avg_epoch_loss:.6f}")
        loss_list.append(avg_epoch_loss)

        Path(config.hme_ckpt).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict()}, config.hme_ckpt)
        print(f"Checkpoint saved: {config.hme_ckpt}")

        # Reconstruction visualization
        model.eval()
        rand_ids = np.random.choice(len(dataset), size=3, replace=False)
        rand_batch = torch.stack([dataset[i] for i in rand_ids]).to(config.device)
        with torch.no_grad():
            output = model(rand_batch.permute(0, 2, 1))
        pred_batch = output["pred"].permute(0, 2, 1).detach()
        x_gt = torch.flatten(rand_batch.permute(0, 2, 1), 0, 1).cpu().numpy()
        x_pred = torch.flatten(pred_batch.permute(0, 2, 1), 0, 1).cpu().numpy()

        norm = mcolors.Normalize(
            vmin=min(float(np.nanmin(x_gt)), float(np.nanmin(x_pred))),
            vmax=max(float(np.nanmax(x_gt)), float(np.nanmax(x_pred))),
        )

        plt.subplot(1, 2, 1)
        plt.xlabel("Ground Truth")
        plt.imshow(x_gt, norm=norm)
        plt.subplot(1, 2, 2)
        plt.imshow(x_pred, norm=norm)
        plt.xlabel(f"Prediction[epoch-{ep_id}]")
        plt.savefig(f"{log_dir}/rec_ep{ep_id:05d}.png")
        plt.close()

        plt.plot(loss_list)
        plt.grid()
        plt.xlabel(f"Loss[epoch-{ep_id}]")
        plt.savefig(f"{log_dir}/loss_ep{ep_id:05d}.png")
        plt.close()

    return loss_list


if __name__ == "__main__":
    train_pae(tyro.cli(PAETrainConfig))
