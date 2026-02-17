import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from main import SpatialAudioHeatmapLocator
from dataset import load_dataset


def train(dataset_dir="dataset",
          epochs=50,
          batch_size=16,
          lr=1e-3,
          azi_bins=72,
          dist_bins=20,
          device=None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- 1. Load dataset ---
    chunks, labels = load_dataset(dataset_dir)
    num_samples = chunks.shape[0]
    print(f"Loaded {num_samples} samples — chunks {chunks.shape}, labels {labels.shape}")

    # Convert to tensors
    chunks_t = torch.from_numpy(chunks)   # (N, 2, n_mels, T)
    labels_t = torch.from_numpy(labels)   # (N, azi_bins, dist_bins)

    # --- 2. Model, optimizer ---
    model = SpatialAudioHeatmapLocator(
        input_channels=2, azi_bins=azi_bins, dist_bins=dist_bins
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- 3. Training loop ---
    model.train()
    for epoch in tqdm(range(epochs), desc="Epochs"):
        # Shuffle
        perm = torch.randperm(num_samples)
        chunks_t = chunks_t[perm]
        labels_t = labels_t[perm]

        epoch_loss = 0.0
        num_batches = 0

        batch_bar = tqdm(range(0, num_samples, batch_size), desc=f"Epoch {epoch+1}", leave=False)
        for start in batch_bar:
            end = min(start + batch_size, num_samples)
            x = chunks_t[start:end].to(device)
            y = labels_t[start:end].to(device)

            pred = model(x)  # (B, azi_bins, dist_bins)
            loss = F.binary_cross_entropy_with_logits(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            batch_bar.set_postfix(loss=f"{loss.item():.6f}")

        avg_loss = epoch_loss / num_batches
        tqdm.write(f"Epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.6f}")

    # --- 4. Save model ---
    save_path = "model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved model to {save_path}")


if __name__ == "__main__":
    train()
