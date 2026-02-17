import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

from main import SpatialAudioHeatmapLocator
from convert_wav import NUM_FEATURE_CHANNELS
from dataset import load_dataset


def focal_dice_loss(pred_logits, target, alpha=0.75, gamma=2.0, dice_weight=0.5):
    """Combined focal + dice loss for sparse heatmap targets.

    Focal loss downweights easy negatives (the many zeros) so the model
    can't just predict all-zero.  Dice loss measures overlap and is
    inherently balanced.
    """
    # --- Focal loss ---
    bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
    prob = torch.sigmoid(pred_logits)
    p_t = prob * target + (1 - prob) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    focal = alpha_t * (1 - p_t) ** gamma * bce
    focal = focal.mean()

    # --- Dice loss ---
    pred_flat = prob.view(prob.size(0), -1)
    tgt_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * tgt_flat).sum(dim=1)
    dice = 1 - (2 * intersection + 1) / (pred_flat.sum(dim=1) + tgt_flat.sum(dim=1) + 1)
    dice = dice.mean()

    return (1 - dice_weight) * focal + dice_weight * dice


class LiveComparisonPlot:
    """Persistent matplotlib window that updates GT vs Predicted each epoch.

    Runs on the main thread — call .update() after each validation step.
    Uses fig.clf() + full redraw to avoid colorbar bugs on polar axes.
    """

    def __init__(self):
        plt.ion()
        self._fig = plt.figure(figsize=(14, 6))
        self._fig.suptitle("Waiting for first epoch…", fontsize=15)
        self._fig.canvas.draw()
        plt.pause(0.01)

    def update(self, gt, pred_logits, epoch):
        pred_prob = 1.0 / (1.0 + np.exp(-pred_logits))

        azi_bins, dist_bins = gt.shape
        azimuths = np.linspace(0, 2 * np.pi, azi_bins)
        distances = np.linspace(0, dist_bins, dist_bins)
        R, Theta = np.meshgrid(distances, azimuths)

        vmin, vmax = 0.0, max(gt.max(), pred_prob.max())

        # Full clear + redraw (avoids colorbar removal issues)
        self._fig.clf()

        ax_gt = self._fig.add_subplot(1, 2, 1, projection="polar")
        ax_pred = self._fig.add_subplot(1, 2, 2, projection="polar")

        for ax, data, label in [(ax_gt, gt, "Ground Truth"),
                                (ax_pred, pred_prob, "Predicted")]:
            pc = ax.pcolormesh(Theta, R, data, shading="auto", cmap="magma",
                               vmin=vmin, vmax=vmax)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_title(label, fontsize=13, pad=12)
            self._fig.colorbar(pc, ax=ax, label="Probability", shrink=0.8)

        self._fig.suptitle(f"Epoch {epoch} — Random Val Sample", fontsize=15)
        self._fig.tight_layout()
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        plt.pause(0.01)


def train(dataset_dir="dataset",
          epochs=100,
          batch_size=8,
          lr=1e-4,
          azi_bins=36,
          dist_bins=5,
          device=None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- 1. Load dataset ---
    chunks, labels = load_dataset(dataset_dir)
    num_samples = chunks.shape[0]
    print(f"Loaded {num_samples} samples — chunks {chunks.shape}, labels {labels.shape}")

    # Convert to tensors
    chunks_t = torch.from_numpy(chunks)   # (N, C, F, T)
    labels_t = torch.from_numpy(labels)   # (N, azi_bins, dist_bins)

    # --- 2. Train/val split (70/30) — sequential, no shuffle ---
    split = int(num_samples * 0.7)

    train_chunks, train_labels = chunks_t[:split], labels_t[:split]
    val_chunks, val_labels = chunks_t[split:], labels_t[split:]
    print(f"Train: {len(train_chunks)}  Val: {len(val_chunks)}")

    # --- 3. Model, optimizer ---
    model = SpatialAudioHeatmapLocator(
        input_channels=NUM_FEATURE_CHANNELS, azi_bins=azi_bins, dist_bins=dist_bins
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- Live plot (persistent window, main thread) ---
    live_plot = LiveComparisonPlot()

    # --- 4. Training loop ---
    for epoch in tqdm(range(epochs), desc="Epochs"):
        # Shuffle train set
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
            loss = focal_dice_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1
            batch_bar.set_postfix(loss=f"{loss.item():.6f}")

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_batches = 0

        # Shuffle val set
        perm = torch.randperm(len(val_chunks))
        val_chunks = val_chunks[perm]
        val_labels = val_labels[perm]

        with torch.no_grad():
            for start in range(0, len(val_chunks), batch_size):
                end = min(start + batch_size, len(val_chunks))
                x = val_chunks[start:end].to(device)
                y = val_labels[start:end].to(device)

                pred = model(x)
                loss = focal_dice_loss(pred, y)

                val_loss += loss.item()
                val_batches += 1

        avg_train = train_loss / train_batches
        avg_val = val_loss / val_batches
        tqdm.write(f"Epoch {epoch+1:3d}/{epochs}  "
                   f"train_loss={avg_train:.6f}  val_loss={avg_val:.6f}")

        # --- Update live plot (non-blocking, main thread) ---
        idx = random.randint(0, len(val_chunks) - 1)
        with torch.no_grad():
            sample_pred = model(val_chunks[idx].unsqueeze(0).to(device))
        live_plot.update(
            val_labels[idx].numpy(),
            sample_pred.squeeze(0).cpu().numpy(),
            epoch + 1,
        )

    # --- 5. Save model ---
    save_path = "model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved model to {save_path}")


if __name__ == "__main__":
    train()
