import os
import glob
import random
import pickle
import numpy as np
import slab
import librosa
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from scipy.signal import fftconvolve

from convert_wav import (
    DEFAULT_SAMPLE_RATE, DEFAULT_N_FFT, DEFAULT_HOP_LENGTH,
    DEFAULT_N_MELS, DEFAULT_WINDOW_SIZE_SECONDS, DEFAULT_GCC_MAX_TAU,
    DEFAULT_FREQ_BINS, NUM_FEATURE_CHANNELS, compute_spatial_features,
)

# Match slab's HRIR samplerate to our audio pipeline
slab.set_default_samplerate(DEFAULT_SAMPLE_RATE)


def _gaussian_heatmap_1d(target_azi_idx, azi_bins=180, sigma=8.0):
    """Creates a 1D circular gaussian blob on the azimuth axis."""
    azi_range = np.arange(azi_bins, dtype=np.float32)
    d_azi = np.minimum(np.abs(azi_range - target_azi_idx),
                       azi_bins - np.abs(azi_range - target_azi_idx))
    heatmap = np.exp(-(d_azi**2) / (2 * sigma**2))
    return heatmap / heatmap.max()


# ============================================================
# Caches (module-level, persist across epochs)
# ============================================================

_wav_paths_cache = {}        # sounds_dir -> list[str]
_audio_cache = {}            # wav_path -> np.ndarray (samples,)  mono at target sr
_hrir_cache = {}             # (azi_deg_rounded, dist_rounded) -> (hrir_L, hrir_R)  np arrays


def _get_wav_paths(sounds_dir):
    """Get list of wav file paths (cached)."""
    if sounds_dir not in _wav_paths_cache:
        wav_files = glob.glob(os.path.join(sounds_dir, "**", "*.wav"), recursive=True)
        assert len(wav_files) > 0, f"No .wav files found in {sounds_dir}"
        _wav_paths_cache[sounds_dir] = wav_files
    return _wav_paths_cache[sounds_dir]


def preload_audio(sounds_dir, sr=DEFAULT_SAMPLE_RATE):
    """Load all wav files into memory as mono numpy arrays at target sr."""
    wav_files = _get_wav_paths(sounds_dir)
    loaded = 0
    for path in tqdm(wav_files, desc="Loading audio files", leave=False):
        if path not in _audio_cache:
            try:
                sound = slab.Sound.read(path)
                if sound.samplerate != sr:
                    sound = sound.resample(sr)
                # Store as mono (just channel 0, or mean if stereo)
                data = sound.data
                if data.ndim > 1:
                    data = data[:, 0]
                _audio_cache[path] = data.astype(np.float32).ravel()
                loaded += 1
            except Exception:
                pass
    if loaded > 0:
        print(f"Preloaded {loaded} new audio files ({len(_audio_cache)} total in cache)")


HRIR_CACHE_PATH = "hrir_cache.pkl"


def precompute_hrir_grid(azi_step=1, dist_steps=None, max_distance=3.0,
                         room_size=None, listener_pos=None):
    """Precompute HRIRs on a grid and cache them (disk + memory)."""
    global _hrir_cache

    if room_size is None:
        room_size = [10, 10, 3]
    if listener_pos is None:
        listener_pos = [5, 5, 1.5]
    if dist_steps is None:
        dist_steps = np.linspace(0.3, max_distance, 10).tolist()

    total = len(range(0, 360, azi_step)) * len(dist_steps)

    # Try loading from disk first
    if not _hrir_cache and os.path.exists(HRIR_CACHE_PATH):
        print(f"Loading HRIR cache from {HRIR_CACHE_PATH}...")
        with open(HRIR_CACHE_PATH, "rb") as f:
            _hrir_cache = pickle.load(f)
        print(f"Loaded {len(_hrir_cache)} cached HRIRs")

    # Check if we already have everything
    already_cached = sum(1 for a in range(0, 360, azi_step)
                         for d in dist_steps if (a, round(d, 2)) in _hrir_cache)
    if already_cached == total:
        return

    print(f"Precomputing HRIR grid ({360 // azi_step} azimuths × {len(dist_steps)} distances)...")
    room = slab.Room(size=room_size, listener=listener_pos)

    for azi_deg in tqdm(range(0, 360, azi_step), desc="HRIR grid", leave=False):
        for dist_m in dist_steps:
            key = (azi_deg, round(dist_m, 2))
            if key in _hrir_cache:
                continue
            room.set_source([float(azi_deg), 0, float(dist_m)])
            hrir = room.hrir()
            hrir_data = hrir.data  # (n_taps, 2)
            _hrir_cache[key] = (hrir_data[:, 0].copy(), hrir_data[:, 1].copy())

    # Save to disk
    with open(HRIR_CACHE_PATH, "wb") as f:
        pickle.dump(_hrir_cache, f)
    print(f"HRIR cache: {len(_hrir_cache)} entries (saved to {HRIR_CACHE_PATH})")



def _lookup_hrir(azi_deg, dist_m, azi_step=1, dist_steps=None, max_distance=3.0):
    """Find nearest cached HRIR for given azimuth and distance."""
    if dist_steps is None:
        dist_steps = np.linspace(0.3, max_distance, 10)

    # Snap azimuth to nearest grid point
    azi_snapped = round(azi_deg / azi_step) * azi_step % 360

    # Snap distance to nearest grid point
    dist_snapped = round(float(dist_steps[np.argmin(np.abs(dist_steps - dist_m))]), 2)

    return _hrir_cache[(azi_snapped, dist_snapped)]


# ============================================================
# Worker for feature computation (multiprocessing)
# ============================================================

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


# ============================================================
# Main generation function
# ============================================================

def generate_epoch(total_duration_seconds=300,
                   num_sounds=50,
                   window_size_seconds=DEFAULT_WINDOW_SIZE_SECONDS,
                   update_interval_ms=100,
                   sr=DEFAULT_SAMPLE_RATE,
                   n_mels=DEFAULT_N_MELS,
                   n_fft=DEFAULT_N_FFT,
                   hop_length=DEFAULT_HOP_LENGTH,
                   azi_bins=180,
                   max_distance=3.0,
                   sigma=10.0,
                   sounds_dir=r"sounds\sounds",
                   num_workers=14):
    """Generate a fresh random dataset in memory using cached HRIRs and audio.

    Returns:
        chunks: np.ndarray of shape (num_windows, 7, F_max, T)
        labels: np.ndarray of shape (num_windows, azi_bins)
    """
    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 1)

    # Ensure caches are populated
    preload_audio(sounds_dir, sr=sr)
    precompute_hrir_grid(max_distance=max_distance)

    wav_files = _get_wav_paths(sounds_dir)
    # Filter to only files we successfully loaded
    wav_files = [f for f in wav_files if f in _audio_cache]

    # --- 1. Spatialize using cached HRIRs (main thread, fast) ---
    total_samples = int(total_duration_seconds * sr)
    stereo_buffer = np.zeros((2, total_samples), dtype=np.float64)
    events = []      # (start, end, azi_deg)
    sound_data = []   # per-sound spatialized arrays for intensity measurement

    for i in range(num_sounds):
        wav_path = random.choice(wav_files)
        azi_deg = random.uniform(0, 360)
        dist_m = random.uniform(0.3, max_distance)  # still used for realistic HRIR
        max_start = max(0, total_samples - int(5 * sr))
        start_sample = random.randint(0, max_start)

        # Get cached audio and HRIR
        dry_mono = _audio_cache[wav_path]
        hrir_L, hrir_R = _lookup_hrir(azi_deg, dist_m, max_distance=max_distance)

        # Convolve
        spatial_L = fftconvolve(dry_mono, hrir_L, mode="full")
        spatial_R = fftconvolve(dry_mono, hrir_R, mode="full")

        # Place in buffer
        end_sample = min(start_sample + len(spatial_L), total_samples)
        length = end_sample - start_sample
        stereo_buffer[0, start_sample:end_sample] += spatial_L[:length]
        stereo_buffer[1, start_sample:end_sample] += spatial_R[:length]
        events.append((start_sample, end_sample, azi_deg))
        sound_data.append((spatial_L[:length], spatial_R[:length]))

    # Normalize
    peak = np.max(np.abs(stereo_buffer))
    if peak > 0:
        stereo_buffer = stereo_buffer / peak * 0.9

    # --- 2. Compute spatial features in parallel ---
    window_samples = int(window_size_seconds * sr)
    hop_samples = int(sr * update_interval_ms / 1000)
    assert window_samples >= hop_samples

    num_windows = (total_samples - window_samples) // hop_samples + 1

    n_gcc_bins = DEFAULT_GCC_MAX_TAU
    F_max = n_fft // 2 + 1

    dummy_stft = librosa.stft(np.zeros(window_samples, dtype=np.float32),
                              n_fft=n_fft, hop_length=hop_length)
    T = dummy_stft.shape[1]

    spec_jobs = []
    for win_idx in range(num_windows):
        win_start = win_idx * hop_samples
        win_end = win_start + window_samples
        chunk = stereo_buffer[:, win_start:win_end]
        spec_jobs.append((win_idx, chunk[0], chunk[1], sr, n_mels, n_fft, hop_length, n_gcc_bins))

    chunks = np.zeros((num_windows, NUM_FEATURE_CHANNELS, F_max, T), dtype=np.float32)

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        results = list(tqdm(pool.map(_compute_spectrogram, spec_jobs, chunksize=32),
                            total=num_windows, desc="Features", leave=False))

    for win_idx, spec in results:
        chunks[win_idx] = spec

    # --- 3. Labels with intensity-scaled blobs ---
    tolerance_samples = int(0.2 * sr)
    min_overlap_samples = int(0.07 * sr)

    labels = np.zeros((num_windows, azi_bins), dtype=np.float32)
    sound_count = np.zeros(num_windows, dtype=np.int32)

    for win_idx in range(num_windows):
        win_start = win_idx * hop_samples
        win_end = win_start + window_samples
        snapshot = win_end - tolerance_samples

        # First pass: collect RMS of each overlapping sound in this window
        active = []  # (azi_deg, rms)
        for ev_i, (ev_start, ev_end, azi_deg) in enumerate(events):
            overlap = min(ev_end, win_end) - max(ev_start, snapshot)
            if overlap >= min_overlap_samples:
                s_L, s_R = sound_data[ev_i]
                local_start = max(0, snapshot - ev_start)
                local_end = min(len(s_L), win_end - ev_start)
                if local_end > local_start:
                    seg = np.concatenate([s_L[local_start:local_end],
                                          s_R[local_start:local_end]])
                    rms = np.sqrt(np.mean(seg**2))
                else:
                    rms = 0.0
                active.append((azi_deg, rms))

        sound_count[win_idx] = len(active)

        if not active:
            continue

        # Normalize against loudest sound in THIS window
        max_rms = max(r for _, r in active)

        for azi_deg, rms in active:
            intensity = 0.5 + 0.5 * min(rms / (max_rms + 1e-8), 1.0)
            azi_idx = round(azi_deg / 360 * azi_bins) % azi_bins
            blob = _gaussian_heatmap_1d(azi_idx, azi_bins, sigma)
            labels[win_idx] = np.maximum(labels[win_idx], blob * intensity)

    # Keep only windows with exactly 1 active sound
    single_source = sound_count == 1
    chunks = chunks[single_source]
    labels = labels[single_source]

    return chunks, labels


# --- CLI: quick test ---
if __name__ == "__main__":
    chunks, labels = generate_epoch(
        total_duration_seconds=60,
        num_sounds=30,
        update_interval_ms=3000,
    )
    print(f"Chunks: {chunks.shape}")
    print(f"Labels: {labels.shape}")
    print(f"Label range: [{labels.min():.4f}, {labels.max():.4f}]")
