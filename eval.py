"""Evaluate model.pt: generate test data and cycle through GT vs Predicted visualizations."""

import torch
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from main import SpatialAudioHeatmapLocator
from convert_wav import NUM_FEATURE_CHANNELS
from dataset import generate_epoch


def evaluate(model_path="model.pt",
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
    chunks, labels = generate_epoch(
        total_duration_seconds=test_duration,
        num_sounds=num_sounds,
        update_interval_ms=2000,
    )
    chunks = torch.from_numpy(chunks)
    labels_np = labels
    print(f"{len(chunks)} test samples")

    # Setup plot
    plt.ion()
    fig = plt.figure(figsize=(14, 6))

    azimuths = np.linspace(0, 2 * np.pi, azi_bins, endpoint=False)
    width = 2 * np.pi / azi_bins

    idx = 0
    order = np.random.permutation(len(chunks))

    def show_sample(i):
        sample_idx = order[i % len(order)]
        x = chunks[sample_idx].unsqueeze(0).to(device)
        gt = labels_np[sample_idx]

        with torch.no_grad():
            pred_logits = model(x).squeeze(0).cpu().numpy()
        pred_prob = 1.0 / (1.0 + np.exp(-pred_logits))

        fig.clf()

        ax_gt = fig.add_subplot(1, 2, 1, projection="polar")
        ax_pred = fig.add_subplot(1, 2, 2, projection="polar")

        for ax, data, label in [(ax_gt, gt, "Ground Truth"),
                                (ax_pred, pred_prob, "Predicted")]:
            colors = plt.cm.magma(data)
            ax.bar(azimuths, np.ones_like(data), width=width, bottom=0.0,
                   color=colors, alpha=0.9)
            ax.set_ylim(0, 1)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_title(label, fontsize=13, pad=12)

        fig.suptitle(f"Sample {sample_idx} ({i+1}/{len(chunks)})", fontsize=15)
        fig.tight_layout()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    def on_key(event):
        nonlocal idx
        if event.key in ("right", " ", "n"):
            idx += 1
            show_sample(idx)
        elif event.key in ("left", "p"):
            idx = max(0, idx - 1)
            show_sample(idx)
        elif event.key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    show_sample(0)
    print("\nControls: Right/Space = next, Left = prev, Q/Esc = quit")
    plt.show(block=True)


if __name__ == "__main__":
    evaluate()
