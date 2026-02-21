"""Evaluate model.pt: generate test data and cycle through GT vs Predicted visualizations."""

import torch
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from main import SpatialAudioHeatmapLocator
from convert_wav import NUM_FEATURE_CHANNELS
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
        input_channels=NUM_FEATURE_CHANNELS, azi_bins=azi_bins
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded {model_path}")

    # Generate test data
    num_sounds = int(test_duration * 1)
    print(f"Generating test data ({test_duration}s, {num_sounds} sounds)...")
    chunks, labels, metadata = generate_epoch(
        total_duration_seconds=test_duration,
        num_sounds=num_sounds,
        update_interval_ms=2000,
    )
    chunks = torch.from_numpy(chunks)
    labels_np = labels
    print(f"{len(chunks)} test samples")

    # Setup live plot
    live_plot = LiveComparisonPlot()

    idx = 0
    order = np.random.permutation(len(chunks))

    def show_sample(i):
        sample_idx = order[i % len(order)]
        x = chunks[sample_idx].unsqueeze(0).to(device)
        gt = labels_np[sample_idx]
        sample_md = metadata[sample_idx]

        with torch.no_grad():
            pred_logits = model(x).squeeze(0).cpu().numpy()
            
        # Update via LiveComparisonPlot
        live_plot.update(gt, pred_logits, sample_md, f"Sample {sample_idx} ({i+1}/{len(chunks)})")

    print("\nCycling through samples every 2 seconds. Press Ctrl+C to stop.")
    import time
    try:
        for i in range(len(chunks)):
            show_sample(i)
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nStopping evaluation...")
    finally:
        live_plot.stop()


if __name__ == "__main__":
    evaluate()
