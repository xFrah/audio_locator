import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import random
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from main import SpatialAudioHeatmapLocator
from convert_wav import NUM_FEATURE_CHANNELS
from dataset import generate_epoch


def bce_loss(pred_logits, target):
    """BCE with dynamic pos_weight = N/K so predicting all-zero is not free."""
    num_elements = target.numel()
    num_pos = target.sum().clamp(min=1)
    pos_weight = (num_elements / num_pos).clamp(max=100.0)  # cap to avoid explosion
    return F.binary_cross_entropy_with_logits(
        pred_logits, target,
        pos_weight=pos_weight.expand_as(target),
    )


class LiveComparisonPlot:
    """Persistent matplotlib window that updates GT vs Predicted each epoch."""

    def __init__(self):
        plt.ion()
        self._fig = plt.figure(figsize=(14, 6))
        self._fig.suptitle("Waiting for first epoch…", fontsize=15)
        self._fig.canvas.draw()
        plt.pause(0.01)

    def update(self, gt, pred_logits, epoch):
        pred_prob = 1.0 / (1.0 + np.exp(-pred_logits))

        azi_bins = gt.shape[0]
        azimuths = np.linspace(0, 2 * np.pi, azi_bins, endpoint=False)
        width = 2 * np.pi / azi_bins

        self._fig.clf()

        ax_gt = self._fig.add_subplot(1, 2, 1, projection="polar")
        ax_pred = self._fig.add_subplot(1, 2, 2, projection="polar")

        for ax, data, label in [(ax_gt, gt, "Ground Truth"),
                                (ax_pred, pred_prob, "Predicted")]:
            colors = plt.cm.magma(data)
            ax.bar(azimuths, np.ones_like(data), width=width, bottom=0.0,
                   color=colors, alpha=0.9)
            ax.set_ylim(0, 1)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_title(label, fontsize=13, pad=12)

        self._fig.suptitle(f"Epoch {epoch} — Sample", fontsize=15)
        self._fig.tight_layout()
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        plt.pause(0.01)


def train(epochs=100,
          batch_size=16,
          lr=2e-4,
          azi_bins=180,
          epoch_duration_seconds=5000,
          device=None):

    if device is None:
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Model, optimizer ---
    model = SpatialAudioHeatmapLocator(
        input_channels=NUM_FEATURE_CHANNELS, azi_bins=azi_bins
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- Live plot ---
    live_plot = LiveComparisonPlot()

    # --- Prefetch helper ---
    num_sounds = int(epoch_duration_seconds * 1)
    gen_kwargs = dict(total_duration_seconds=epoch_duration_seconds,
                      num_sounds=num_sounds, update_interval_ms=2000)

    prefetch = ThreadPoolExecutor(max_workers=1)

    # Kick off first epoch generation (blocks until ready)
    tqdm.write(f"Generating first epoch data ({epoch_duration_seconds}s, {num_sounds} sounds)…")
    next_train_future = prefetch.submit(generate_epoch, **gen_kwargs)

    # --- Training loop ---
    for epoch in tqdm(range(epochs), desc="Epochs"):

        # Wait for this epoch's data
        train_chunks, train_labels = next_train_future.result()
        train_chunks = torch.from_numpy(train_chunks)
        train_labels = torch.from_numpy(train_labels)
        tqdm.write(f"\n--- Epoch {epoch+1}: {len(train_chunks)} train samples ready")

        # Prefetch NEXT epoch's data while we train
        if epoch + 1 < epochs:
            next_train_future = prefetch.submit(generate_epoch, **gen_kwargs)

        # Shuffle
        perm = torch.randperm(len(train_chunks))
        train_chunks = train_chunks[perm]
        train_labels = train_labels[perm]

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_batches = 0

        batch_bar = tqdm(range(0, len(train_chunks), batch_size),
                         desc=f"Epoch {epoch+1} train", leave=False)
        for start in batch_bar:
            end = min(start + batch_size, len(train_chunks))
            x = train_chunks[start:end].to(device)
            y = train_labels[start:end].to(device)

            pred = model(x)
            loss = bce_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1
            batch_bar.set_postfix(loss=f"{loss.item():.6f}")

            # Update live plot with last sample of this batch
            live_plot.update(
                y[-1].cpu().numpy(),
                pred[-1].detach().cpu().numpy(),
                epoch + 1,
            )

        avg_train = train_loss / train_batches
        tqdm.write(f"Epoch {epoch+1:3d}/{epochs}  train_loss={avg_train:.6f}")

    # --- Save model ---
    save_path = "model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved model to {save_path}")


if __name__ == "__main__":
    train()

