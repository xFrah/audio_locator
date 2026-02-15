import torch
import torch.nn as nn
import torchaudio
from torchaudio.models import Conformer
from scipy.optimize import linear_sum_assignment


class SpatialAudioTransformer(nn.Module):
    def __init__(self, input_channels=2, embed_dim=256, n_heads=8):
        super().__init__()

        self.embed_dim = embed_dim

        # --- ENCODER: ResNet + Conformer ---
        self.front_end = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Freq 128 -> 64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Freq 64 -> 32
        )
        self.enc_proj = nn.Linear(128 * 32, embed_dim)
        self.conformer = Conformer(
            input_dim=embed_dim,
            ffn_dim=embed_dim * 4,  # This was the missing piece!
            num_heads=n_heads,
            num_layers=4,
            depthwise_conv_kernel_size=31,
        )
        # --- DECODER ---
        # Projects 3D coords + 1 EOE bit back to latent space for autoregression
        self.coord_embedder = nn.Linear(4, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=n_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

        # Output: [Azimuth, Elevation, Distance, EOE_Probability]
        self.fc_out = nn.Linear(embed_dim, 4)
        self.sos_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x, target_seq=None):
        b = x.shape[0]

        # Encode
        x = self.front_end(x)
        x = x.permute(0, 3, 1, 2).reshape(b, -1, 128 * 32)
        memory, _ = self.conformer(self.enc_proj(x), torch.full((b,), x.size(1), device=x.device))

        if self.training:
            assert target_seq is not None, "target_seq is not set while training."
            # Teacher Forcing: Project target coords to embedding space
            # target_seq shape: (B, Num_Sources, 4)
            sos = self.sos_token.expand(b, -1, -1)
            target_embedded = self.coord_embedder(target_seq)
            decoder_input = torch.cat([sos, target_embedded], dim=1)[:, :-1, :]

            out = self.decoder(decoder_input, memory)
            return self.fc_out(out)

        else:
            # Inference: Recursive generation
            results = []
            curr_input = self.sos_token.expand(b, -1, -1)

            for _ in range(10):  # Max sources
                out = self.decoder(curr_input, memory)
                pred = self.fc_out(out[:, -1:, :])
                results.append(pred)

                # Check EOE (if 4th value > 0.5)
                if torch.sigmoid(pred[0, 0, 3]) > 0.5:
                    break

                next_emb = self.coord_embedder(pred)
                curr_input = torch.cat([curr_input, next_emb], dim=1)

            return torch.cat(results, dim=1)


# --- Hungarian Loss Implementation ---
def spatial_hungarian_loss(predictions, targets):
    """
    predictions: (Batch, N, 4) -> [A, E, D, EOE]
    targets: (Batch, M, 4)
    """
    batch_size = predictions.size(0)
    total_loss = 0

    # We use MSE for the first 3 (coords) and BCE for the 4th (EOE)
    mse = nn.MSELoss(reduction="none")
    bce = nn.BCEWithLogitsLoss(reduction="none")

    for b in range(batch_size):
        pred_coords = predictions[b, :, :3]  # (N, 3)
        true_coords = targets[b, :, :3]  # (M, 3)

        # 1. Compute Cost Matrix (Distance between every pred and every truth)
        # Using simple Euclidean distance for the matching cost
        cost_matrix = torch.cdist(pred_coords, true_coords).cpu().detach().numpy()

        # 2. Hungarian Matching (find optimal pairs)
        row_idx, col_idx = linear_sum_assignment(cost_matrix)

        # 3. Calculate Loss for matched pairs
        coord_loss = mse(pred_coords[row_idx], true_coords[col_idx]).mean()
        eoe_loss = bce(predictions[b, :, 3], targets[b, :, 3]).mean()

        total_loss += coord_loss + eoe_loss

    return total_loss / batch_size


# --- Runtime Example ---
model = SpatialAudioTransformer()
# Fake 2-channel MelSpec: (B=1, C=2, F=128, T=100)
audio_input = torch.randn(1, 2, 128, 100)
# Fake Ground Truth: 2 sources + 1 EOE token
# Each source: [Azim, Elev, Dist, EOE_Flag]
gt_sources = torch.tensor([[[0.5, 0.1, 2.0, 0.0], [-0.5, 0.0, 1.5, 0.0], [0.0, 0.0, 0.0, 1.0]]])

# Forward pass
output = model(audio_input, target_seq=gt_sources)
loss = spatial_hungarian_loss(output, gt_sources)

print(f"Loss value: {loss.item():.4f}")
