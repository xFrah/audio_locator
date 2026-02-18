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
    def __init__(self, input_channels=NUM_FEATURE_CHANNELS, azi_bins=180):
        super().__init__()
        self.azi_bins = azi_bins

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

        # 4. AZIMUTH HEAD
        self.head = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, azi_bins),
        )

    def forward(self, x):
        # x: (B, C, F, T)
        x = self.encoder(x)              # (B, 256, 1, T)
        x = x.squeeze(2).permute(0, 2, 1)  # (B, T, 256)
        rnn_out, _ = self.rnn(x)         # (B, T, 512)

        pooled = self.attn_pool(rnn_out)  # (B, 512)

        logits = self.head(pooled)
        return logits  # (B, azi_bins)

def visualize_azimuth(logits_tensor, title="Azimuth Prediction"):
    """
    logits_tensor: torch.Tensor of shape (azi_bins,)
    """
    data = torch.sigmoid(logits_tensor).detach().cpu().numpy()
    azi_bins = data.shape[0]
    azimuths = np.linspace(0, 2 * np.pi, azi_bins, endpoint=False)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    width = 2 * np.pi / azi_bins
    ax.bar(azimuths, data, width=width, bottom=0.0, color=plt.cm.magma(data), alpha=0.9)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    plt.title(title)
    plt.show()


# --- RUNTIME EXAMPLE ---

if __name__ == "__main__":
    model = SpatialAudioHeatmapLocator(input_channels=NUM_FEATURE_CHANNELS, azi_bins=180)

    real_audio = prepare_audio_input_librosa(r"output\gunshot_azi75_dist1.wav")
    print(f"Input shape: {real_audio.shape}")

    print("Running forward pass...")
    model.eval()
    with torch.no_grad():
        raw_output = model(real_audio)  # Output shape: (1, 36)

    print("Visualizing raw (untrained) weights...")
    visualize_azimuth(raw_output.squeeze(0))