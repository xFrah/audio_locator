import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from convert_wav import prepare_audio_input_librosa, NUM_FEATURE_CHANNELS


class AttentionPool(nn.Module):
    """Attention-weighted pooling over the temporal dimension."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):  # x: (B, T, D)
        weights = torch.softmax(self.attn(x), dim=1)  # (B, T, 1)
        return (weights * x).sum(dim=1)  # (B, D)


class SpatialAudioHeatmapLocator(nn.Module):
    def __init__(self, input_channels=NUM_FEATURE_CHANNELS, azi_bins=36, dist_bins=5):
        super().__init__()
        self.azi_bins = azi_bins
        self.dist_bins = dist_bins

        # 1. ENCODER
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)),  # Collapse freq → 1, keep time
        )

        # 2. TEMPORAL: Bidirectional GRU
        self.rnn = nn.GRU(input_size=256, hidden_size=256,
                          num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.1)

        # 3. ATTENTION POOLING
        self.attn_pool = AttentionPool(256 * 2)

        # 4. HEATMAP HEAD
        self.heatmap_head = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, azi_bins * dist_bins),
        )

    def forward(self, x):
        # x: (B, C, F, T)
        x = self.encoder(x)              # (B, 256, 1, T)
        x = x.squeeze(2).permute(0, 2, 1)  # (B, T, 256)
        rnn_out, _ = self.rnn(x)         # (B, T, 512)

        pooled = self.attn_pool(rnn_out)  # (B, 512)

        logits = self.heatmap_head(pooled)
        return logits.view(-1, self.azi_bins, self.dist_bins)

def visualize_heatmap(heatmap_tensor, title="3D Audio Heatmap"):
    """
    heatmap_tensor: torch.Tensor of shape (azi_bins, dist_bins)
    """
    # Convert to numpy for plotting
    data = torch.sigmoid(heatmap_tensor).detach().cpu().numpy()
    
    azi_bins, dist_bins = data.shape
    
    # Create coordinate grids
    # Azimuth: 0 to 2*pi
    # Distance: 0 to max_bins
    azimuths = np.linspace(0, 2 * np.pi, azi_bins)
    distances = np.linspace(0, dist_bins, dist_bins)
    
    R, Theta = np.meshgrid(distances, azimuths)

    # Plotting
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    
    # pcolormesh creates the "Heat" map
    pc = ax.pcolormesh(Theta, R, data, shading='auto', cmap='magma')
    
    ax.set_theta_zero_location("N") # 0 degrees at the top
    ax.set_theta_direction(-1)     # Clockwise
    
    plt.colorbar(pc, ax=ax, label='Probability Intensity')
    plt.title(title)
    plt.show()

# --- 2D GAUSSIAN SMOOTHING ---


def get_2d_gaussian_heatmap(target_azi_idx, target_dist_idx, azi_bins=36, dist_bins=5, sigma=1.2):
    """Creates a 2D 'blob' of probability on the radar map."""
    azi_range = torch.arange(azi_bins).float()
    dist_range = torch.arange(dist_bins).float()

    # Grid of coordinates
    azi_grid, dist_grid = torch.meshgrid(azi_range, dist_range, indexing="ij")

    # Calculate distance from target (Circular for Azimuth)
    d_azi = torch.min(torch.abs(azi_grid - target_azi_idx), azi_bins - torch.abs(azi_grid - target_azi_idx))
    d_dist = torch.abs(dist_grid - target_dist_idx)

    # 2D Gaussian formula
    heatmap = torch.exp(-(d_azi**2 + d_dist**2) / (2 * sigma**2))
    return heatmap / heatmap.max()


def heatmap_loss(pred_heatmap, target_sources, azi_bins=36, dist_bins=5):
    """
    target_sources: List of (azi_idx, dist_idx) for one batch item
    This allows training with multiple sources in the same clip.
    """
    batch_size = pred_heatmap.size(0)
    total_loss = 0

    for i in range(batch_size):
        # Build the ground truth heatmap by combining blobs for each source
        gt_heatmap = torch.zeros(azi_bins, dist_bins).to(pred_heatmap.device)
        for azi_idx, dist_idx in target_sources[i]:
            blob = get_2d_gaussian_heatmap(azi_idx, dist_idx, azi_bins, dist_bins)
            gt_heatmap = torch.max(gt_heatmap, blob.to(pred_heatmap.device))

        total_loss += F.binary_cross_entropy_with_logits(pred_heatmap[i], gt_heatmap)

    return total_loss / batch_size


# --- RUNTIME EXAMPLE ---

if __name__ == "__main__":
    # 1. Initialize with random weights
    model = SpatialAudioHeatmapLocator(input_channels=NUM_FEATURE_CHANNELS, azi_bins=36, dist_bins=5)
    
    # 2. Load real audio
    real_audio = prepare_audio_input_librosa(r"output\gunshot_azi75_dist1.wav")
    print(f"Input shape: {real_audio.shape}")  # [1, 7, 1025, T]

    print("Running forward pass...")
    # 3. Model Inference
    model.eval()
    with torch.no_grad():
        raw_output = model(real_audio)  # Output shape: (1, 36, 5)
    
    # 4. Strip the batch dimension and visualize
    print("Visualizing raw (untrained) weights...")
    visualize_heatmap(raw_output.squeeze(0))