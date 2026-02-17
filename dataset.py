import os
import glob
import random
import numpy as np
import slab
import librosa

from convert_wav import (
    DEFAULT_SAMPLE_RATE, DEFAULT_N_FFT, DEFAULT_HOP_LENGTH,
    DEFAULT_N_MELS, DEFAULT_WINDOW_SIZE_SECONDS,
)


def _gaussian_heatmap(target_azi_idx, target_dist_idx,
                      azi_bins=72, dist_bins=20, sigma=1.2):
    """Creates a 2D gaussian blob on the heatmap grid (numpy version)."""
    azi_range = np.arange(azi_bins, dtype=np.float32)
    dist_range = np.arange(dist_bins, dtype=np.float32)
    azi_grid, dist_grid = np.meshgrid(azi_range, dist_range, indexing="ij")

    # Circular distance for azimuth
    d_azi = np.minimum(np.abs(azi_grid - target_azi_idx),
                       azi_bins - np.abs(azi_grid - target_azi_idx))
    d_dist = np.abs(dist_grid - target_dist_idx)

    heatmap = np.exp(-(d_azi**2 + d_dist**2) / (2 * sigma**2))
    return heatmap / heatmap.max()


def generate_dataset(total_duration_seconds=300,
                     num_sounds=50,
                     window_size_seconds=DEFAULT_WINDOW_SIZE_SECONDS,
                     update_interval_ms=100,
                     sr=DEFAULT_SAMPLE_RATE,
                     n_mels=DEFAULT_N_MELS,
                     n_fft=DEFAULT_N_FFT,
                     hop_length=DEFAULT_HOP_LENGTH,
                     azi_bins=72,
                     dist_bins=20,
                     max_distance=3.0,
                     sigma=1.2,
                     sounds_dir=r"sounds\sounds",
                     output_dir="dataset",
                     seed=None):
    """Generate audio in memory, slice into windows, compute mel spectrograms,
    and save chunks + labels as .npy.

    Returns:
        chunks_path: Path to saved spectrograms .npy — shape (num_windows, 2, n_mels, T).
        labels_path: Path to saved labels .npy — shape (num_windows, azi_bins, dist_bins).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Collect all source sounds ---
    wav_files = glob.glob(os.path.join(sounds_dir, "**", "*.wav"), recursive=True)
    assert len(wav_files) > 0, f"No .wav files found in {sounds_dir}"
    print(f"Found {len(wav_files)} source sounds")

    # --- 2. Setup slab environment ---
    room_size = [10, 10, 3]
    listener_pos = [5, 5, 1.5]

    total_samples = int(total_duration_seconds * sr)
    stereo_buffer = np.zeros((2, total_samples), dtype=np.float64)

    # --- 3. Place sounds and record events ---
    events = []

    for i in range(num_sounds):
        wav_path = random.choice(wav_files)
        try:
            dry_sound = slab.Sound.read(wav_path)
        except Exception as e:
            print(f"  Skipping {wav_path}: {e}")
            continue

        # Random spatial position
        azi_deg = random.uniform(0, 360)
        dist_m = random.uniform(0.3, max_distance)

        # Random start time (ensure it fits)
        dry_samples = dry_sound.n_samples
        max_start = total_samples - dry_samples
        if max_start <= 0:
            dry_sound = dry_sound.resize(total_samples)
            dry_samples = total_samples
            max_start = 0

        start_sample = random.randint(0, max(0, max_start))

        # Spatialize using slab Room + HRTF
        room = slab.Room(size=room_size, listener=listener_pos)
        room.set_source([azi_deg, 0, dist_m])
        hrir = room.hrir()

        if dry_sound.samplerate != hrir.samplerate:
            dry_sound = dry_sound.resample(hrir.samplerate)

        spatial_audio = hrir.apply(dry_sound)

        if spatial_audio.samplerate != sr:
            spatial_audio = spatial_audio.resample(sr)

        # Mix into the buffer
        end_sample = min(start_sample + spatial_audio.n_samples, total_samples)
        length = end_sample - start_sample
        spatial_data = spatial_audio.data[:length].T  # (2, length)
        stereo_buffer[:, start_sample:end_sample] += spatial_data

        events.append((start_sample, end_sample, azi_deg, dist_m))
        print(f"  [{i+1}/{num_sounds}] azi={azi_deg:.1f}° dist={dist_m:.2f}m "
              f"t={start_sample/sr:.2f}-{end_sample/sr:.2f}s")

    # Normalize to prevent clipping
    peak = np.max(np.abs(stereo_buffer))
    if peak > 0:
        stereo_buffer = stereo_buffer / peak * 0.9

    # --- 4. Slice into windows and compute spectrograms + labels ---
    window_samples = int(window_size_seconds * sr)
    hop_samples = int(sr * update_interval_ms / 1000)
    assert window_samples >= hop_samples, (
        f"Window ({window_size_seconds}s) must be >= hop ({update_interval_ms}ms)")

    num_windows = (total_samples - window_samples) // hop_samples + 1
    print(f"\nSlicing into {num_windows} windows ({window_size_seconds}s each, "
          f"{update_interval_ms}ms hop)...")

    # Pre-compute T for the spectrogram shape
    dummy_spec = librosa.feature.melspectrogram(
        y=np.zeros(window_samples, dtype=np.float32), sr=sr,
        n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    T = dummy_spec.shape[1]

    chunks = np.zeros((num_windows, 2, n_mels, T), dtype=np.float32)
    labels = np.zeros((num_windows, azi_bins, dist_bins), dtype=np.float32)

    for win_idx in range(num_windows):
        win_start = win_idx * hop_samples
        win_end = win_start + window_samples
        audio_chunk = stereo_buffer[:, win_start:win_end]

        # Mel spectrogram per channel
        for ch in range(2):
            S = librosa.feature.melspectrogram(
                y=audio_chunk[ch].astype(np.float32), sr=sr,
                n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
            )
            chunks[win_idx, ch] = librosa.power_to_db(S, ref=np.max)

        # Label: combine gaussian blobs for all active events
        for (ev_start, ev_end, azi_deg, dist_m) in events:
            if ev_start < win_end and ev_end > win_start:
                azi_idx = round(azi_deg / 360 * azi_bins) % azi_bins
                dist_idx = min(round(dist_m / max_distance * (dist_bins - 1)),
                               dist_bins - 1)
                blob = _gaussian_heatmap(azi_idx, dist_idx,
                                         azi_bins, dist_bins, sigma)
                labels[win_idx] = np.maximum(labels[win_idx], blob)

    # --- 5. Save ---
    chunks_path = os.path.join(output_dir, "chunks.npy")
    labels_path = os.path.join(output_dir, "labels.npy")
    np.save(chunks_path, chunks)
    np.save(labels_path, labels)
    print(f"Saved chunks: {chunks_path} — shape {chunks.shape}")
    print(f"Saved labels: {labels_path} — shape {labels.shape}")

    return chunks_path, labels_path

def load_dataset(dataset_dir="dataset"):
    """Load pre-generated chunks and labels from disk.

    Returns:
        chunks: np.ndarray of shape (num_windows, 2, n_mels, T)
        labels: np.ndarray of shape (num_windows, azi_bins, dist_bins)
    """
    chunks = np.load(os.path.join(dataset_dir, "chunks.npy"))
    labels = np.load(os.path.join(dataset_dir, "labels.npy"))
    return chunks, labels


# --- CLI: generate + verify ---
if __name__ == "__main__":
    chunks_path, labels_path = generate_dataset(
        total_duration_seconds=60,
        num_sounds=100,
        update_interval_ms=100,
        seed=42,
    )

    chunks = np.load(chunks_path)
    labels = np.load(labels_path)
    print(f"\nChunks: {chunks.shape}")   # (num_windows, 2, 128, T)
    print(f"Labels: {labels.shape}")     # (num_windows, 72, 20)
    print(f"Label range: [{labels.min():.4f}, {labels.max():.4f}]")
