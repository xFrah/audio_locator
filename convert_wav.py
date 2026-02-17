import torch
import librosa
import numpy as np

# --- Shared Audio Constants ---
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512
DEFAULT_N_MELS = 128
DEFAULT_WINDOW_SIZE_SECONDS = 3.0
DEFAULT_GCC_MAX_TAU = 64  # Number of lag bins to keep from GCC-PHAT

# Derived constants
DEFAULT_FREQ_BINS = DEFAULT_N_FFT // 2 + 1  # 1025
NUM_FEATURE_CHANNELS = 7  # mel_L, mel_R, ILD, stft_mag_L, stft_mag_R, IPD, GCC-PHAT


def compute_t_frames(window_size_seconds=DEFAULT_WINDOW_SIZE_SECONDS,
                     sr=DEFAULT_SAMPLE_RATE,
                     hop_length=DEFAULT_HOP_LENGTH):
    """Compute the number of time frames T for a given window size."""
    total_samples = int(window_size_seconds * sr)
    return total_samples // hop_length + 1


def compute_spatial_features(y_left, y_right, sr=DEFAULT_SAMPLE_RATE,
                             n_fft=DEFAULT_N_FFT, hop_length=DEFAULT_HOP_LENGTH,
                             n_mels=DEFAULT_N_MELS, n_gcc_bins=DEFAULT_GCC_MAX_TAU):
    """Compute all spatial audio features from a stereo pair.

    Returns:
        features: np.ndarray of shape (7, F_max, T) where F_max = n_fft//2+1.
        Channels:
            0 — mel L (dB), zero-padded from n_mels to F_max
            1 — mel R (dB), zero-padded from n_mels to F_max
            2 — ILD (mel_L_dB - mel_R_dB), zero-padded from n_mels to F_max
            3 — STFT magnitude L (dB)
            4 — STFT magnitude R (dB)
            5 — IPD (interaural phase difference)
            6 — GCC-PHAT, zero-padded from n_gcc_bins to F_max
    """
    F_max = n_fft // 2 + 1  # 1025

    # --- Mel spectrograms (n_mels, T) ---
    mel_L = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y_left, sr=sr, n_mels=n_mels,
                                       n_fft=n_fft, hop_length=hop_length),
        ref=np.max)
    mel_R = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y_right, sr=sr, n_mels=n_mels,
                                       n_fft=n_fft, hop_length=hop_length),
        ref=np.max)

    # --- ILD (n_mels, T) ---
    ild = mel_L - mel_R

    # --- STFT (complex) ---
    stft_L = librosa.stft(y_left, n_fft=n_fft, hop_length=hop_length)   # (F_max, T)
    stft_R = librosa.stft(y_right, n_fft=n_fft, hop_length=hop_length)  # (F_max, T)

    T = stft_L.shape[1]

    # STFT magnitude in dB (F_max, T)
    mag_L = librosa.amplitude_to_db(np.abs(stft_L), ref=np.max)
    mag_R = librosa.amplitude_to_db(np.abs(stft_R), ref=np.max)

    # --- IPD (F_max, T) ---
    ipd = np.angle(stft_L * np.conj(stft_R))

    # --- GCC-PHAT (n_gcc_bins, T) ---
    cross_spec = stft_L * np.conj(stft_R)
    denom = np.abs(cross_spec) + 1e-8
    gcc_full = np.fft.irfft(cross_spec / denom, n=n_fft, axis=0)  # (n_fft, T)
    # Keep only the relevant lags around zero: [-n_gcc_bins//2 .. n_gcc_bins//2-1]
    half = n_gcc_bins // 2
    gcc = np.concatenate([gcc_full[-half:, :], gcc_full[:half, :]], axis=0)  # (n_gcc_bins, T)

    # --- Pad everything to (F_max, T) ---
    def pad_freq(x, target_f):
        """Zero-pad along frequency axis."""
        if x.shape[0] >= target_f:
            return x[:target_f]
        pad = np.zeros((target_f - x.shape[0], x.shape[1]), dtype=x.dtype)
        return np.concatenate([x, pad], axis=0)

    features = np.stack([
        pad_freq(mel_L, F_max),   # 0: mel L
        pad_freq(mel_R, F_max),   # 1: mel R
        pad_freq(ild, F_max),     # 2: ILD
        mag_L,                     # 3: STFT mag L  (already F_max)
        mag_R,                     # 4: STFT mag R
        ipd,                       # 5: IPD
        pad_freq(gcc, F_max),     # 6: GCC-PHAT
    ], axis=0).astype(np.float32)  # (7, F_max, T)

    return features


def prepare_audio_input_librosa(file_path,
                                window_size_seconds=DEFAULT_WINDOW_SIZE_SECONDS,
                                sr=DEFAULT_SAMPLE_RATE,
                                n_mels=DEFAULT_N_MELS,
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
                                        hop_length=hop_length, n_mels=n_mels)

    # 4. Convert to Torch Tensor with batch dimension
    input_tensor = torch.from_numpy(features).unsqueeze(0)  # (1, 7, F_max, T)

    return input_tensor


# --- Usage ---
if __name__ == "__main__":
    input_data = prepare_audio_input_librosa(r"output\gunshot_azi75_dist1.wav")
    print(f"Shape: {input_data.shape}")  # [1, 7, 1025, T]
    print(f"T frames: {input_data.shape[-1]} (expected {compute_t_frames()})")
