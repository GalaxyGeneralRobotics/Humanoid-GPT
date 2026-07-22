import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PeriodicAutoencoder(nn.Module):
    def __init__(
        self,
        inp_ch: int,
        latent_ch: int,
        win_len: int,
        hidden_dims: int = (64, 64),
        win_sec: float = 2.0,
    ):
        super().__init__()
        self.inp_ch = inp_ch
        self.win_len = win_len
        self.latent_ch = latent_ch

        kernel_size = int(win_len)
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for 'same' padding.")
        padding = kernel_size // 2

        # Precomputed constants
        self.register_buffer(
            "time_vec",
            torch.linspace(-win_sec / 2, win_sec / 2, win_len),
            persistent=False,
        )
        self.register_buffer(
            "freqs",
            torch.fft.rfftfreq(win_len)[1:] * win_len / win_sec,
            persistent=False,
        )
        self.register_buffer(
            "two_pi",
            torch.tensor(2.0 * math.pi, dtype=torch.float32),
            persistent=False,
        )

        layers = []
        channels = [self.inp_ch, *hidden_dims, latent_ch]
        for in_ch, out_ch in zip(channels[:-1], channels[1:], strict=False):
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding),
                nn.BatchNorm1d(out_ch),
                nn.ELU(),
            ]
        self.encoder = nn.Sequential(*layers)

        # Encoder
        self.phase_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.win_len, 2),
                    nn.BatchNorm1d(2),
                )
                for _ in range(latent_ch)
            ]
        )

        # Decoder
        layers = []
        channels = [latent_ch, *hidden_dims, self.inp_ch]
        for in_ch, out_ch in zip(channels[:-1], channels[1:], strict=False):
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding),
                nn.BatchNorm1d(out_ch),
                nn.ELU(),
            ]
        layers.append(nn.Conv1d(out_ch, inp_ch, kernel_size, padding=padding))
        self.decoder = nn.Sequential(*layers)

    def _compute_fft_params(self, latent: torch.Tensor):
        """
        Computes amplitude, frequency, offset using FFT of latent representation.
        """
        rfft = torch.fft.rfft(latent, dim=-1)
        spectrum = torch.abs(rfft[:, :, 1:])  # skip DC
        power = spectrum**2

        pow_sum = power.sum(dim=-1).add(1e-8)
        w_freq = (self.freqs * power).sum(dim=-1)

        freq = w_freq / pow_sum
        amp = 2 * torch.sqrt(pow_sum) / self.win_len
        offset = rfft.real[:, :, 0] / self.win_len  # DC

        return amp.unsqueeze(-1), freq.unsqueeze(-1), offset.unsqueeze(-1)

    def encode(self, inp: torch.Tensor):
        latent = self.encoder(inp)
        amp, freq, offset = self._compute_fft_params(latent)

        phase_shift_list = []
        for i in range(self.latent_ch):
            enc = self.phase_encoders[i]
            z_shift = enc(latent[:, i])
            p_shift = torch.atan2(z_shift[..., 1], z_shift[..., 0]) / self.two_pi
            phase_shift_list.append(p_shift)

        phase_shift = torch.stack(phase_shift_list, dim=1)
        return {
            "latent": latent,
            "amp": amp,
            "freq": freq,
            "offset": offset,
            "shift": phase_shift.unsqueeze(-1),
        }

    def forward(self, inp: torch.Tensor):
        params = self.encode(inp)
        # latent = params["latent"]
        amp = params["amp"]
        freq = params["freq"]
        offset = params["offset"]
        shift = params["shift"]
        recon_latent = (
            amp * torch.sin(self.two_pi * (freq * self.time_vec + shift)) + offset
        )
        pred = self.decoder(recon_latent)
        loss = F.mse_loss(pred, inp)
        return {"pred": pred, "loss": loss}

    def encode_phase_manifold(self, inp: torch.Tensor):
        encoded_param = self.encode(inp)
        amp = encoded_param["amp"]
        shift = encoded_param["shift"]
        phase_embed = torch.stack(
            [
                amp * torch.sin(self.two_pi * shift),
                amp * torch.cos(self.two_pi * shift),
            ],
            dim=2,
        ).squeeze(-1)
        phase_embed = phase_embed.flatten(start_dim=1)
        encoded_param["manifold"] = phase_embed
        return encoded_param


# ==============================================================================
#                                USAGE EXAMPLE
# ==============================================================================
if __name__ == "__main__":
    # --- Configuration ---
    batch_size = 32
    inp_dim = 30
    window_len = 29
    window_sec = 2.0

    # --- Model Initialization ---
    model = PeriodicAutoencoder(
        inp_ch=inp_dim,
        latent_ch=8,
        win_len=window_len,
        win_sec=window_sec,
    )

    print("✅ Model Initialized Successfully")
    print(f"Time window duration: {window_sec:.2f}s")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}\n")

    # --- Dummy Data ---
    input_motion = torch.randn(batch_size, inp_dim, window_len)
    print(f"Input shape:  {input_motion.shape}")

    # --- Forward Pass & Loss Calculation ---
    try:
        out = model(input_motion)
        loss = out["loss"]
        pred_motion = out["pred"]
        print(f"Output shape: {pred_motion.shape}")
        print(f"Example MSE Loss: {loss.item():.6f}")
        assert input_motion.shape == pred_motion.shape
        print("\n✅ Forward pass successful and shapes match.")
    except Exception as e:
        print(f"\n❌ An error occurred during the forward pass: {e}")
