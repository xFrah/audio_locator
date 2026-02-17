import torch
import librosa
import numpy as np

# --- Shared Audio Constants ---
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512
DEFAULT_N_MELS = 128
DEFAULT_WINDOW_SIZE_SECONDS = 3.0


def compute_t_frames(window_size_seconds=DEFAULT_WINDOW_SIZE_SECONDS,
                     sr=DEFAULT_SAMPLE_RATE,
                     hop_length=DEFAULT_HOP_LENGTH):
    """Compute the number of time frames T for a given window size."""
    total_samples = int(window_size_seconds * sr)
    return total_samples // hop_length + 1


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

    # 3. Compute Mel Spectrogram for each channel
    specs = []
    for i in range(2):
        S = librosa.feature.melspectrogram(y=y[i], sr=sr, n_mels=n_mels,
                                           n_fft=n_fft, hop_length=hop_length)
        S_db = librosa.power_to_db(S, ref=np.max)
        specs.append(S_db)

    # 4. Stack into [2, n_mels, T] and convert to Torch Tensor
    combined_spec = np.stack(specs)
    input_tensor = torch.from_numpy(combined_spec).unsqueeze(0)  # Add Batch dim

    return input_tensor


# --- Usage ---
if __name__ == "__main__":
    input_data = prepare_audio_input_librosa(r"output\gunshot_azi75_dist1.wav")
    print(f"Shape: {input_data.shape}")  # [1, 2, 128, T]
    print(f"T frames: {input_data.shape[-1]} (expected {compute_t_frames()})")
