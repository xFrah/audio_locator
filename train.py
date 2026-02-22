import torch
import warnings
import os
import threading
import time
import gc

# Suppress "PyTorch is not compiled with NCCL support" warning on Windows
warnings.filterwarnings("ignore", message=".*PyTorch is not compiled with NCCL support.*")
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

from main import SpatialAudioHeatmapLocator
from convert_wav import NUM_SPATIAL_CHANNELS, NUM_GCC_CHANNELS
from dataset import generate_epoch
from plot import LiveComparisonPlot


def bce_loss(pred_logits, target):
    """BCE with dynamic pos_weight = N/K so predicting all-zero is not free."""
    num_elements = target.numel()
    num_pos = target.sum().clamp(min=1)
    pos_weight = (num_elements / num_pos).clamp(max=100.0)  # cap to avoid explosion
    return F.binary_cross_entropy_with_logits(
        pred_logits,
        target,
        pos_weight=pos_weight.expand_as(target),
    )


def train(epochs=100, batch_size=24, lr=1e-5, azi_bins=180, epoch_duration_seconds=5000, device=None):

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Model, optimizer ---
    model = SpatialAudioHeatmapLocator(input_channels=NUM_SPATIAL_CHANNELS, gcc_channels=NUM_GCC_CHANNELS, azi_bins=azi_bins).to(device)

    if os.path.exists("resume.pt"):
        print("Found resume.pt, attempting to resume training from these weights...")
        try:
            state_dict = torch.load("resume.pt", map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            print("Successfully loaded resume.pt")
        except RuntimeError as e:
            print(f"Failed to load resume.pt due to architecture change. Starting fresh. Error snippet: {str(e)[:100]}...")

    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- Live plot ---
    live_plot = LiveComparisonPlot()

    # --- Data generation config ---
    num_sounds = int(epoch_duration_seconds * 1)
    gen_kwargs = dict(total_duration_seconds=epoch_duration_seconds, num_sounds=num_sounds, update_interval_ms=2000, max_velocity=90, moving_prob=0.8)

    # --- Training loop ---
    best_loss = float("inf")

    for epoch in tqdm(range(epochs), desc="Epochs"):

        # Generate this epoch's data
        tqdm.write(f"\nGenerating epoch {epoch+1} data ({epoch_duration_seconds}s, {num_sounds} sounds)…")
        train_chunks_spatial, train_chunks_gcc, train_labels, train_metadata, _ = generate_epoch(**gen_kwargs)
        train_chunks_spatial = torch.from_numpy(train_chunks_spatial)
        train_chunks_gcc = torch.from_numpy(train_chunks_gcc)
        train_labels = torch.from_numpy(train_labels)
        tqdm.write(f"--- Epoch {epoch+1}: {len(train_chunks_spatial)} train samples ready")

        # Shuffle
        perm = torch.randperm(len(train_chunks_spatial))
        train_chunks_spatial = train_chunks_spatial[perm]
        train_chunks_gcc = train_chunks_gcc[perm]
        train_labels = train_labels[perm]
        train_metadata = [train_metadata[i] for i in perm.tolist()]

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_batches = 0

        batch_bar = tqdm(range(0, len(train_chunks_spatial), batch_size), desc=f"Epoch {epoch+1} train", leave=False)
        for start in batch_bar:
            end = min(start + batch_size, len(train_chunks_spatial))
            x_spatial = train_chunks_spatial[start:end].to(device)
            x_gcc = train_chunks_gcc[start:end].to(device)
            y = train_labels[start:end].to(device)

            pred = model(x_spatial, x_gcc)
            loss = bce_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1
            batch_bar.set_postfix(loss=f"{loss.item():.6f}")

            # Update live plot with last sample of this batch
            # Update every batch as requested by user
            live_plot.update(
                y[-1].cpu().numpy(),
                pred[-1].detach().cpu().numpy(),
                train_metadata[end - 1],  # Sample metadata
                epoch + 1,
            )

        avg_train = train_loss / train_batches
        tqdm.write(f"Epoch {epoch+1:3d}/{epochs}  train_loss={avg_train:.6f}")

        # Save if best
        if avg_train < best_loss:
            best_loss = avg_train
            save_path = "model.pt"
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state_dict, save_path)
            tqdm.write(f"New best loss! Saved model to {save_path}")

        # --- Explicit Garbage Collection to prevent RAM spikes ---
        del train_chunks_spatial
        del train_chunks_gcc
        del train_labels
        del train_metadata
        gc.collect()

    live_plot.stop()
    print(f"\nTraining complete. Best loss: {best_loss:.6f}")


if __name__ == "__main__":
    train()
