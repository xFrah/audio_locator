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

    # --- 2. Train/val split (70/30) ---
    split = int(num_samples * 0.7)
    perm = torch.randperm(num_samples)
    train_idx, val_idx = perm[:split], perm[split:]

    train_chunks, train_labels = chunks_t[train_idx], labels_t[train_idx]
    val_chunks, val_labels = chunks_t[val_idx], labels_t[val_idx]
    print(f"Train: {len(train_chunks)}  Val: {len(val_chunks)}")

    # --- 3. Model, optimizer ---
    model = SpatialAudioHeatmapLocator(
        input_channels=2, azi_bins=azi_bins, dist_bins=dist_bins
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
            loss = F.binary_cross_entropy_with_logits(pred, y)

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
                loss = F.binary_cross_entropy_with_logits(pred, y)

                val_loss += loss.item()
                val_batches += 1

        avg_train = train_loss / train_batches
        avg_val = val_loss / val_batches
        tqdm.write(f"Epoch {epoch+1:3d}/{epochs}  "
                   f"train_loss={avg_train:.6f}  val_loss={avg_val:.6f}")

    # --- 4. Save model ---
    save_path = "model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved model to {save_path}")


if __name__ == "__main__":
    train()
