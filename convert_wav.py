import torch
import librosa
import numpy as np

# --- Shared Audio Constants ---
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512
DEFAULT_WINDOW_SIZE_SECONDS = 2.0
DEFAULT_GCC_MAX_TAU = 64  # Number of lag bins to keep from GCC-PHAT

# Derived constants
DEFAULT_FREQ_BINS = DEFAULT_N_FFT // 2 + 1  # 1025
NUM_FEATURE_CHANNELS = 5  # stft_mag_L, stft_mag_R, ILD, IPD, GCC-PHAT


def compute_t_frames(window_size_seconds=DEFAULT_WINDOW_SIZE_SECONDS,
                     sr=DEFAULT_SAMPLE_RATE,
                     hop_length=DEFAULT_HOP_LENGTH):
    """Compute the number of time frames T for a given window size."""
    total_samples = int(window_size_seconds * sr)
    return total_samples // hop_length + 1


def compute_spatial_features(y_left, y_right, sr=DEFAULT_SAMPLE_RATE,
                             n_fft=DEFAULT_N_FFT, hop_length=DEFAULT_HOP_LENGTH,
                             n_gcc_bins=DEFAULT_GCC_MAX_TAU):
    """Compute all spatial audio features from a stereo pair.

    Returns:
        features: np.ndarray of shape (5, F_max, T) where F_max = n_fft//2+1.
        Channels:
            0 — STFT magnitude L (dB)
            1 — STFT magnitude R (dB)
            2 — ILD (mag_L_dB - mag_R_dB), full STFT resolution
            3 — IPD (interaural phase difference)
            4 — GCC-PHAT, zero-padded from n_gcc_bins to F_max
    """
    F_max = n_fft // 2 + 1  # 1025

    # --- STFT (complex) ---
    stft_L = librosa.stft(y_left, n_fft=n_fft, hop_length=hop_length)   # (F_max, T)
    stft_R = librosa.stft(y_right, n_fft=n_fft, hop_length=hop_length)  # (F_max, T)

    T = stft_L.shape[1]

    # STFT magnitude in dB (F_max, T)
    mag_L = librosa.amplitude_to_db(np.abs(stft_L), ref=np.max)
    mag_R = librosa.amplitude_to_db(np.abs(stft_R), ref=np.max)

    # --- ILD at full STFT resolution (F_max, T) ---
    ild = mag_L - mag_R

    # --- IPD (F_max, T) ---
    ipd = np.angle(stft_L * np.conj(stft_R))

    # --- GCC-PHAT (n_gcc_bins, T) ---
    cross_spec = stft_L * np.conj(stft_R)
    denom = np.abs(cross_spec) + 1e-8
    gcc_full = np.fft.irfft(cross_spec / denom, n=n_fft, axis=0)  # (n_fft, T)
    # Keep only the relevant lags around zero: [-n_gcc_bins//2 .. n_gcc_bins//2-1]
    half = n_gcc_bins // 2
    gcc = np.concatenate([gcc_full[-half:, :], gcc_full[:half, :]], axis=0)  # (n_gcc_bins, T)

    # --- Pad GCC to (F_max, T) ---
    def pad_freq(x, target_f):
        """Zero-pad along frequency axis."""
        if x.shape[0] >= target_f:
            return x[:target_f]
        pad = np.zeros((target_f - x.shape[0], x.shape[1]), dtype=x.dtype)
        return np.concatenate([x, pad], axis=0)

    features = np.stack([
        mag_L,                     # 0: STFT mag L
        mag_R,                     # 1: STFT mag R
        ild,                       # 2: ILD (full resolution)
        ipd,                       # 3: IPD
        pad_freq(gcc, F_max),     # 4: GCC-PHAT
    ], axis=0).astype(np.float32)  # (5, F_max, T)

    return features


def prepare_audio_input_librosa(file_path,
                                window_size_seconds=DEFAULT_WINDOW_SIZE_SECONDS,
                                sr=DEFAULT_SAMPLE_RATE,
                                n_fft=DEFAULT_N_FFT,
                                hop_length=DEFAULT_HOP_LENGTH):
    # 1. Load audio (mono=False keeps it stereo)
    y, _ = librosa.load(file_path, sr=sr, mono=False)

    assert y.shape[0] == 2, "Audio file must have exactly 2 channels."

    # 2. Trim or pad to exactly window_size_seconds
    target_samples = int(window_size_seconds * sr)
    if y.shape[1] > target_samples:
        y = y[:, :target_samples]
    elif y.shape[1] < target_samples:
        pad_width = target_samples - y.shape[1]
        y = np.pad(y, ((0, 0), (0, pad_width)), mode='constant')

    # 3. Compute spatial features
    features = compute_spatial_features(y[0], y[1], sr=sr, n_fft=n_fft,
                                        hop_length=hop_length)

    # 4. Convert to Torch Tensor with batch dimension
    input_tensor = torch.from_numpy(features).unsqueeze(0)  # (1, 5, F_max, T)

    return input_tensor


# --- Usage ---
if __name__ == "__main__":
    input_data = prepare_audio_input_librosa(r"output\gunshot_azi75_dist1.wav")
    print(f"Shape: {input_data.shape}")  # [1, 7, 1025, T]
    print(f"T frames: {input_data.shape[-1]} (expected {compute_t_frames()})")
