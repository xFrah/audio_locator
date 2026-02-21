import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from convert_wav import NUM_SPATIAL_CHANNELS, NUM_GCC_CHANNELS


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


class CoordConv2d(nn.Module):
    """
    Adds Y (frequency) coordinate channel to give absolute spatial awareness
    before applying standard Conv2d.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        # +1 for the Y-coord channel (ignoring X-coord which is time and shifts)
        self.conv = nn.Conv2d(in_channels + 1, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        b, c, h, w = x.shape
        # Create a linear spread [-1, 1] across the H dimension (Freq)
        y_coords = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
        y_coords = y_coords.view(1, 1, h, 1).expand(b, 1, h, w)
        x = torch.cat([x, y_coords], dim=1)
        return self.conv(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for dynamic channel weighting."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEResNetBlock(nn.Module):
    """Residual Block with Squeeze-and-Excitation and larger temporal stride/dilation support."""
    def __init__(self, in_channels, out_channels, stride=(2, 1), kernel_size=(3, 3)):
        super().__init__()
        
        # Temporal receptive field expansion via padding
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.se = SEBlock(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or stride != (1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        
        out += identity
        out = F.relu(out)
        return out


class SpatialAudioHeatmapLocator(nn.Module):
    def __init__(self, input_channels=NUM_SPATIAL_CHANNELS, gcc_channels=NUM_GCC_CHANNELS, azi_bins=180):
        super().__init__()
        self.azi_bins = azi_bins

        # 1. ENCODER FOR SPATIAL FEATURES (ResNet + CoordConv + SE Blocks)
        # Input: (B, 4, 1025, T)
        
        # Initial CoordConv preserves the critical frequency mappings immediately
        self.init_conv = nn.Sequential(
            CoordConv2d(input_channels, 64, kernel_size=(5, 5), stride=(2, 1), padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.encoder = nn.Sequential(
            # 513 -> 257  (Using wider temporal kernel to catch ITDs/ILDs early across frames)
            SEResNetBlock(64, 128, stride=(2, 1), kernel_size=(3, 5)),
            # 257 -> 129
            SEResNetBlock(128, 256, stride=(2, 1), kernel_size=(3, 5)),
            # 129 -> 65
            SEResNetBlock(256, 512, stride=(2, 1), kernel_size=(3, 3)),
            # 65 -> 33
            SEResNetBlock(512, 1024, stride=(2, 1), kernel_size=(3, 3)),
            # 33 -> 17
            SEResNetBlock(1024, 1024, stride=(2, 1), kernel_size=(3, 3)),
            # Stop downsampling freq here to preserve HRTF spectral cues (17 bins)
            SEResNetBlock(1024, 1024, stride=(1, 1), kernel_size=(3, 3)),
        )

        # Dimension reduction to save parameters before RNN
        self.spatial_proj = nn.Conv2d(1024, 256, kernel_size=1)
        
        # 1.5. ENCODER FOR GCC-PHAT FEATURES
        # Input: (B, 1, 64, T)
        self.gcc_conv = nn.Sequential(
            nn.Conv2d(gcc_channels, 32, kernel_size=(3, 5), stride=(2, 1), padding=(1, 2)), # 64 -> 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            SEResNetBlock(32, 64, stride=(2, 1), kernel_size=(3, 3)),   # 32 -> 16
            SEResNetBlock(64, 128, stride=(2, 1), kernel_size=(3, 3)),  # 16 -> 8
            SEResNetBlock(128, 256, stride=(2, 1), kernel_size=(3, 3)), # 8 -> 4
        )

        # Output shapes: Spatial = (B, 256, 17, T) -> Flatten freq -> (B, 4352, T)
        #                GCC = (B, 256, 4, T)     -> Flatten lag -> (B, 1024, T)
        rnn_input_dim = (256 * 17) + (256 * 4) # 5376

        # 2. TEMPORAL: Bidirectional GRU
        self.rnn = nn.GRU(input_size=rnn_input_dim, hidden_size=1024,
                          num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.1)

        # 3. ATTENTION POOLING
        self.attn_pool = AttentionPool(1024 * 2)

        # 4. AZIMUTH HEAD
        self.head = nn.Sequential(
            nn.Linear(1024 * 2, 1024),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(1024, azi_bins)
        )

    def forward(self, x_spatial, x_gcc):
        # x_spatial: (B, 4, F, T)
        # x_gcc: (B, 1, L, T)
        
        # Spatial branch
        s = self.init_conv(x_spatial)    # (B, 64, 513, T)
        s = self.encoder(s)              # (B, 1024, 17, T)
        s = self.spatial_proj(s)         # (B, 256, 17, T)
        
        B, C_s, F_s, T = s.shape
        s = s.permute(0, 3, 1, 2).contiguous().view(B, T, C_s * F_s)  # (B, T, 4352)
        
        # GCC branch
        g = self.gcc_conv(x_gcc)         # (B, 256, 4, T)
        B, C_g, F_g, T = g.shape
        g = g.permute(0, 3, 1, 2).contiguous().view(B, T, C_g * F_g)  # (B, T, 1024)
        
        # Concatenate features along feature dimension
        x = torch.cat([s, g], dim=-1)    # (B, T, 5376)
        
        rnn_out, _ = self.rnn(x)         # (B, T, 2048)

        pooled = self.attn_pool(rnn_out)  # (B, 2048)

        logits = self.head(pooled)       # (B, azi_bins)
        return logits


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
    model = SpatialAudioHeatmapLocator(input_channels=NUM_SPATIAL_CHANNELS, gcc_channels=NUM_GCC_CHANNELS, azi_bins=180)

    # Use dummy input
    dummy_spatial = torch.randn(1, NUM_SPATIAL_CHANNELS, 1025, 431) 
    dummy_gcc = torch.randn(1, NUM_GCC_CHANNELS, 64, 431) 
    print(f"Dummy spatial shape: {dummy_spatial.shape}")
    print(f"Dummy gcc shape: {dummy_gcc.shape}")

    print("Running forward pass...")
    model.eval()
    with torch.no_grad():
        raw_output = model(dummy_spatial, dummy_gcc)  # Output shape: (1, 180)

    print(f"Output shape: {raw_output.shape}")
    print("Visualizing raw (untrained) weights...")
    visualize_azimuth(raw_output.squeeze(0))