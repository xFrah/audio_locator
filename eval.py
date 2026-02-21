"""Evaluate model.pt: generate test data and cycle through GT vs Predicted visualizations."""

import torch
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from main import SpatialAudioHeatmapLocator
from convert_wav import NUM_SPATIAL_CHANNELS, NUM_GCC_CHANNELS
from dataset import generate_epoch
from train import LiveComparisonPlot


def evaluate(model_path="resume.pt",
             test_duration=300,
             azi_bins=180,
             device=None):

    if device is None:
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    model = SpatialAudioHeatmapLocator(
        input_channels=NUM_SPATIAL_CHANNELS, gcc_channels=NUM_GCC_CHANNELS, azi_bins=azi_bins
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded {model_path}")
    except Exception as e:
        print(f"Failed to load {model_path}, using random weights. Error snippet: {str(e)[:100]}...")
    model.eval()

    # Generate test data
    num_sounds = int(test_duration * 1)
    print(f"Generating test data ({test_duration}s, {num_sounds} sounds)...")
    chunks_spatial, chunks_gcc, labels, metadata = generate_epoch(
        total_duration_seconds=test_duration,
        num_sounds=num_sounds,
        update_interval_ms=2000,
    )
    chunks_spatial = torch.from_numpy(chunks_spatial)
    chunks_gcc = torch.from_numpy(chunks_gcc)
    labels_np = labels
    print(f"{len(chunks_spatial)} test samples")

    # Setup live plot
    live_plot = LiveComparisonPlot()

    idx = 0
    order = np.random.permutation(len(chunks_spatial))

    def show_sample(i):
        sample_idx = order[i % len(order)]
        x_spatial = chunks_spatial[sample_idx].unsqueeze(0).to(device)
        x_gcc = chunks_gcc[sample_idx].unsqueeze(0).to(device)
        gt = labels_np[sample_idx]
        sample_md = metadata[sample_idx]

        with torch.no_grad():
            pred_logits = model(x_spatial, x_gcc).squeeze(0).cpu().numpy()
            
        # Update via LiveComparisonPlot
        live_plot.update(gt, pred_logits, sample_md, f"Sample {sample_idx} ({i+1}/{len(chunks_spatial)})")

    print("\nCycling through samples every 2 seconds. Press Ctrl+C to stop.")
    import time
    try:
        for i in range(len(chunks_spatial)):
            show_sample(i)
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nStopping evaluation...")
    finally:
        live_plot.stop()


if __name__ == "__main__":
    evaluate()
