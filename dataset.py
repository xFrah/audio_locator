import os
import glob
import random
import numpy as np
import slab
import librosa
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from convert_wav import (
    DEFAULT_SAMPLE_RATE, DEFAULT_N_FFT, DEFAULT_HOP_LENGTH,
    DEFAULT_N_MELS, DEFAULT_WINDOW_SIZE_SECONDS, DEFAULT_GCC_MAX_TAU,
    DEFAULT_FREQ_BINS, NUM_FEATURE_CHANNELS, compute_spatial_features,
)


def _gaussian_heatmap(target_azi_idx, target_dist_idx,
                      azi_bins=36, dist_bins=5, sigma=1.2):
    """Creates a 2D gaussian blob on the heatmap grid (numpy version)."""
    azi_range = np.arange(azi_bins, dtype=np.float32)
    dist_range = np.arange(dist_bins, dtype=np.float32)
    azi_grid, dist_grid = np.meshgrid(azi_range, dist_range, indexing="ij")

    d_azi = np.minimum(np.abs(azi_grid - target_azi_idx),
                       azi_bins - np.abs(azi_grid - target_azi_idx))
    d_dist = np.abs(dist_grid - target_dist_idx)

    heatmap = np.exp(-(d_azi**2 + d_dist**2) / (2 * sigma**2))
    return heatmap / heatmap.max()


# --- Worker functions for multiprocessing ---

def _spatialize_sound(args):
    """Worker: load, spatialize, and return result as numpy array."""
    wav_path, azi_deg, dist_m, start_sample, total_samples, sr = args
    room_size = [10, 10, 3]
    listener_pos = [5, 5, 1.5]

    try:
        dry_sound = slab.Sound.read(wav_path)
    except Exception:
        return None

    # Truncate if sound is longer than the buffer
    if dry_sound.n_samples > total_samples:
        dry_sound = dry_sound.resize(total_samples)

    room = slab.Room(size=room_size, listener=listener_pos)
    room.set_source([azi_deg, 0, dist_m])
    hrir = room.hrir()

    if dry_sound.samplerate != hrir.samplerate:
        dry_sound = dry_sound.resample(hrir.samplerate)

    spatial_audio = hrir.apply(dry_sound)

    if spatial_audio.samplerate != sr:
        spatial_audio = spatial_audio.resample(sr)

    end_sample = min(start_sample + spatial_audio.n_samples, total_samples)
    length = end_sample - start_sample
    spatial_data = spatial_audio.data[:length].T  # (2, length)

    return (start_sample, end_sample, azi_deg, dist_m, spatial_data)


def _compute_spectrogram(args):
    """Worker: compute spatial features for one window."""
    win_idx, audio_chunk_ch0, audio_chunk_ch1, sr, n_mels, n_fft, hop_length, n_gcc_bins = args

    features = compute_spatial_features(
        audio_chunk_ch0.astype(np.float32),
        audio_chunk_ch1.astype(np.float32),
        sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, n_gcc_bins=n_gcc_bins,
    )  # (7, F_max, T)

    return win_idx, features


def generate_dataset(total_duration_seconds=300,
                     num_sounds=50,
                     window_size_seconds=DEFAULT_WINDOW_SIZE_SECONDS,
                     update_interval_ms=100,
                     sr=DEFAULT_SAMPLE_RATE,
                     n_mels=DEFAULT_N_MELS,
                     n_fft=DEFAULT_N_FFT,
                     hop_length=DEFAULT_HOP_LENGTH,
                     azi_bins=36,
                     dist_bins=5,
                     max_distance=3.0,
                     sigma=1.2,
                     sounds_dir=r"sounds\sounds",
                     output_dir="dataset",
                     num_workers=14,
                     seed=None):
    """Generate audio in memory, slice into windows, compute mel spectrograms,
    and save chunks + labels as .npy. Uses multiprocessing for speed.

    Returns:
        chunks_path: Path to saved spectrograms .npy — shape (num_windows, 2, n_mels, T).
        labels_path: Path to saved labels .npy — shape (num_windows, azi_bins, dist_bins).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 1)

    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Collect all source sounds ---
    wav_files = glob.glob(os.path.join(sounds_dir, "**", "*.wav"), recursive=True)
    assert len(wav_files) > 0, f"No .wav files found in {sounds_dir}"
    print(f"Found {len(wav_files)} source sounds, using {num_workers} workers")

    # --- 2. Prepare spatialization jobs ---
    total_samples = int(total_duration_seconds * sr)

    jobs = []
    for i in range(num_sounds):
        wav_path = random.choice(wav_files)
        azi_deg = random.uniform(0, 360)
        dist_m = random.uniform(0.3, max_distance)
        # Estimate start_sample (conservative: assume max 5s sound)
        max_start = max(0, total_samples - int(5 * sr))
        start_sample = random.randint(0, max_start)
        jobs.append((wav_path, azi_deg, dist_m, start_sample, total_samples, sr))

    # --- 3. Spatialize in parallel ---
    print(f"Spatializing {num_sounds} sounds...")
    stereo_buffer = np.zeros((2, total_samples), dtype=np.float64)
    events = []

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        results = list(tqdm(pool.map(_spatialize_sound, jobs),
                            total=len(jobs), desc="HRTF"))

    for result in results:
        if result is None:
            continue
        start_sample, end_sample, azi_deg, dist_m, spatial_data = result
        stereo_buffer[:, start_sample:end_sample] += spatial_data
        events.append((start_sample, end_sample, azi_deg, dist_m))

    print(f"Placed {len(events)}/{num_sounds} sounds")

    # Normalize
    peak = np.max(np.abs(stereo_buffer))
    if peak > 0:
        stereo_buffer = stereo_buffer / peak * 0.9

    # --- 4. Compute spatial features in parallel ---
    window_samples = int(window_size_seconds * sr)
    hop_samples = int(sr * update_interval_ms / 1000)
    assert window_samples >= hop_samples, (
        f"Window ({window_size_seconds}s) must be >= hop ({update_interval_ms}ms)")

    num_windows = (total_samples - window_samples) // hop_samples + 1
    print(f"\nComputing spatial features for {num_windows} windows...")

    n_gcc_bins = DEFAULT_GCC_MAX_TAU
    F_max = n_fft // 2 + 1  # 1025

    # Pre-compute T from a dummy STFT
    dummy_stft = librosa.stft(np.zeros(window_samples, dtype=np.float32),
                              n_fft=n_fft, hop_length=hop_length)
    T = dummy_stft.shape[1]

    # Build spectrogram jobs
    spec_jobs = []
    for win_idx in range(num_windows):
        win_start = win_idx * hop_samples
        win_end = win_start + window_samples
        chunk = stereo_buffer[:, win_start:win_end]
        spec_jobs.append((win_idx, chunk[0], chunk[1], sr, n_mels, n_fft, hop_length, n_gcc_bins))

    chunks = np.zeros((num_windows, NUM_FEATURE_CHANNELS, F_max, T), dtype=np.float32)

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        results = list(tqdm(pool.map(_compute_spectrogram, spec_jobs, chunksize=32),
                            total=num_windows, desc="Features"))

    for win_idx, spec in results:
        chunks[win_idx] = spec

    # --- 5. Labels (fast, no parallelism needed) ---
    print("Computing labels...")
    labels = np.zeros((num_windows, azi_bins, dist_bins), dtype=np.float32)

    for win_idx in range(num_windows):
        win_start = win_idx * hop_samples
        win_end = win_start + window_samples

        for (ev_start, ev_end, azi_deg, dist_m) in events:
            if ev_start < win_end and ev_end > win_start:
                azi_idx = round(azi_deg / 360 * azi_bins) % azi_bins
                dist_idx = min(round(dist_m / max_distance * (dist_bins - 1)),
                               dist_bins - 1)
                blob = _gaussian_heatmap(azi_idx, dist_idx,
                                         azi_bins, dist_bins, sigma)
                labels[win_idx] = np.maximum(labels[win_idx], blob)

    # --- 6. Save ---
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

    total_duration = 15000
    chunks_path, labels_path = generate_dataset(
        total_duration_seconds=total_duration,
        num_sounds=int(total_duration * 0.7),
        update_interval_ms=3000,
        seed=42,
    )

    chunks = np.load(chunks_path)
    labels = np.load(labels_path)
    print(f"\nChunks: {chunks.shape}")   # (num_windows, 7, 1025, T)
    print(f"Labels: {labels.shape}")     # (num_windows, 36, 5)
    print(f"Label range: [{labels.min():.4f}, {labels.max():.4f}]")
