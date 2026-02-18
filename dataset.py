import os
import glob
import random
import pickle
import numpy as np
import slab
import librosa
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from scipy.signal import fftconvolve

from convert_wav import (
    DEFAULT_SAMPLE_RATE, DEFAULT_N_FFT, DEFAULT_HOP_LENGTH,
    DEFAULT_WINDOW_SIZE_SECONDS, DEFAULT_GCC_MAX_TAU,
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
    global _hrir_cache
    if not _hrir_cache:
        # Reload cache in worker process
        if os.path.exists(HRIR_CACHE_PATH):
            with open(HRIR_CACHE_PATH, "rb") as f:
                _hrir_cache = pickle.load(f)
        else:
            raise RuntimeError("HRIR cache not found in worker process. Ensure precompute_hrir_grid() ran in main process.")

    if dist_steps is None:
        dist_steps = np.linspace(0.3, max_distance, 10)

    # Snap azimuth to nearest grid point
    azi_snapped = round(azi_deg / azi_step) * azi_step % 360

    # Snap distance to nearest grid point
    dist_snapped = round(float(dist_steps[np.argmin(np.abs(dist_steps - dist_m))]), 2)

    return _hrir_cache[(azi_snapped, dist_snapped)]


def _get_interpolated_hrir(azi_deg, dist_m, max_distance=3.0):
    """Get HRIR, supporting float queries by snapping (nearest neighbor for now)."""
    # In the future, we could bilinearly interpolate between grid points.
    # For now, nearest neighbor is sufficient if the grid is fine enough (1 deg).
    return _lookup_hrir(azi_deg, dist_m, max_distance=max_distance)



# ============================================================
# Worker for feature computation (multiprocessing)
# ============================================================

def _compute_spectrogram(args):
    """Worker: compute spatial features for one window from shared memory."""
    win_idx, win_start, win_end, shm_name, buf_shape, sr, n_fft, hop_length, n_gcc_bins = args

    # Attach to shared memory and read window slice
    shm = shared_memory.SharedMemory(name=shm_name)
    buffer = np.ndarray(buf_shape, dtype=np.float64, buffer=shm.buf)
    ch0 = buffer[0, win_start:win_end].astype(np.float32)
    ch1 = buffer[1, win_start:win_end].astype(np.float32)
    shm.close()

    features = compute_spatial_features(
        ch0, ch1,
        sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_gcc_bins=n_gcc_bins,
    )  # (5, F_max, T)

    return win_idx, features


def _spatialize_sound(args):
    """Worker: convolve mono sound with HRIR and add to shared memory buffer."""
    (idx, wav_path, start_azi, start_dist, end_azi, end_dist, max_distance,
     start_sample, shm_name, buf_shape, seed) = args

    # Reseed to ensure different randomness if needed (though we pass explicit params here)
    np.random.seed(seed)

    # 1. Get cached audio and HRIR (read-only from module cache)
    # Note: _audio_cache and _hrir_cache are available because of copy-on-write fork
    # on Linux/Mac, but on Windows we might need to reload if not using 'fork'
    # However, for now we rely on the fact that _lookup_hrir uses the module-level _hrir_cache
    # If _hrir_cache is empty in worker, we might need to reload it.
    # On Windows 'spawn', module globals are re-imported.
    # _audio_cache is populated by preload_audio, which we might need to call if empty.

    # Simpler: just reload/compute HRIR here if missing is fast, but better to rely on args
    # actually, passing large arrays in args is bad.
    # Let's rely on _audio_cache being populated.
    # On Windows, we need to ensure caches are ready in the worker.
    # But since we use 'spawn', the module is imported fresh.
    # We must explicitly reload caches in worker if they are empty?
    # Actually, let's just use the main process for looking up arrays and pass them?
    # No, passing 5000 arrays is slow.
    # best way: cache is in module scope.
    # For now, let's assume valid cache or quick reload.

    # WAIT: on Windows "spawn", the new process imports the module but doesn't share global state.
    # So _audio_cache will be EMPTY in the worker unless we populate it.
    # Populating it takes time.
    #
    # ALTERNATIVE: Use the main thread for spatialization if cache sharing is hard?
    # OR: Pass the specific mono audio array and HRIR filters in args?
    # Mono audio is small (<1MB). HRIR is tiny. Passing them in args is fine!

    dry_mono = wav_path  # args can contain the data directly
    n_samples = len(dry_mono)

    # Check if stationary (or close enough)
    is_stationary = (abs(start_azi - end_azi) < 0.5) and (abs(start_dist - end_dist) < 0.1)

    if is_stationary:
        # --- Fast Path: Static Convolution ---
        hrir_L, hrir_R = _lookup_hrir(start_azi, start_dist, max_distance=max_distance)
        spatial_L = fftconvolve(dry_mono, hrir_L, mode="full")
        spatial_R = fftconvolve(dry_mono, hrir_R, mode="full")
    
    else:
        # --- Dynamic Path: Block-based Overlap-Add ---
        block_size = 2048
        # HRIR length (assumed constant)
        dummy_L, _ = _lookup_hrir(0, 1.0, max_distance=max_distance)
        hrir_len = len(dummy_L)
        
        out_len = n_samples + hrir_len - 1
        spatial_L = np.zeros(out_len, dtype=np.float32)
        spatial_R = np.zeros(out_len, dtype=np.float32)

        for b_start in range(0, n_samples, block_size):
            b_end = min(b_start + block_size, n_samples)
            block = dry_mono[b_start:b_end]
            
            # Calculate position at center of block
            # (Simple linear interpolation over time)
            t_center = (b_start + b_end) / 2 / n_samples
            curr_azi = start_azi + (end_azi - start_azi) * t_center
            curr_dist = start_dist + (end_dist - start_dist) * t_center
            
            # Lookup HRIR for this block
            h_L, h_R = _get_interpolated_hrir(curr_azi, curr_dist, max_distance=max_distance)
            
            # Convolve
            # mode='full' gives length N + M - 1
            conv_L = fftconvolve(block, h_L, mode="full")
            conv_R = fftconvolve(block, h_R, mode="full")
            
            # Overlap-add
            l_conv = len(conv_L)
            target_end = min(b_start + l_conv, len(spatial_L))
            write_len = target_end - b_start
            
            if write_len > 0:
                spatial_L[b_start : target_end] += conv_L[:write_len]
                spatial_R[b_start : target_end] += conv_R[:write_len]

    return idx, start_sample, spatial_L, spatial_R


# ============================================================
# Main generation function
# ============================================================

_initialized = False
_filtered_wav_files = []
_cached_T = None
_persistent_pool = None
_stereo_buffer_template = None  # pre-allocated zeros buffer
_shm = None  # persistent shared memory block


def generate_epoch(total_duration_seconds=300,
                   num_sounds=50,
                   window_size_seconds=DEFAULT_WINDOW_SIZE_SECONDS,
                   update_interval_ms=100,
                   sr=DEFAULT_SAMPLE_RATE,
                   n_fft=DEFAULT_N_FFT,
                   hop_length=DEFAULT_HOP_LENGTH,
                   azi_bins=180,
                   max_distance=3.0,
                   sigma=15.0,
                   sounds_dir=r"sounds\sounds",
                   num_workers=14,
                   moving_prob=0.5,       # Probability that a sound moves
                   max_velocity=90.0):    # Max velocity in degrees/second
    """Generate a fresh random dataset in memory using cached HRIRs and audio.

    Returns:
        chunks: np.ndarray of shape (num_windows, NUM_FEATURE_CHANNELS, F_max, T)
        labels: np.ndarray of shape (num_windows, azi_bins)
    """
    global _initialized, _filtered_wav_files, _cached_T, _persistent_pool, _shm

    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 1)



    # --- One-time initialization ---
    if not _initialized:
        preload_audio(sounds_dir, sr=sr)
        precompute_hrir_grid(max_distance=max_distance)

        wav_files = _get_wav_paths(sounds_dir)
        _filtered_wav_files = [f for f in wav_files if f in _audio_cache]

        # Precompute T (number of STFT time frames)
        window_samples = int(window_size_seconds * sr)
        dummy_stft = librosa.stft(np.zeros(window_samples, dtype=np.float32),
                                  n_fft=n_fft, hop_length=hop_length)
        _cached_T = dummy_stft.shape[1]

        # Start persistent process pool
        _persistent_pool = ProcessPoolExecutor(max_workers=num_workers)

        _initialized = True
        print(f"Initialization complete: {len(_filtered_wav_files)} audio files, T={_cached_T}")

    wav_files = _filtered_wav_files
    T = _cached_T


    # --- Ensure shared memory block exists for this epoch size ---
    total_samples = int(total_duration_seconds * sr)
    buf_shape = (2, total_samples)
    buf_nbytes = int(np.prod(buf_shape)) * 8  # float64

    if _shm is not None:
        try:
            _shm.close()
            _shm.unlink()
        except Exception:
            pass
    _shm = shared_memory.SharedMemory(create=True, size=buf_nbytes)
    stereo_buffer = np.ndarray(buf_shape, dtype=np.float64, buffer=_shm.buf)
    stereo_buffer[:] = 0


    # --- 1. Spatialize in parallel (Batched to save RAM) ---
    # We process sounds in batches to avoid creating a massive list of jobs that consumes RAM
    # (15k sounds * 1MB per sound = 15GB list overhead)
    batch_size = 1000  # Process 1000 sounds at a time
    events = []        # (start, end, azi_deg)
    
    # Pre-allocate list for sound data (will be filled by index)
    # We still need this for intensity calculation, but it stores just the small arrays
    sound_data_list = [None] * num_sounds

    overall_pbar = tqdm(total=num_sounds, desc="Spatializing", leave=False)

    for batch_start in range(0, num_sounds, batch_size):
        batch_end = min(batch_start + batch_size, num_sounds)
        
        spat_jobs = []
        batch_indices = range(batch_start, batch_end)
        
        for i in batch_indices:
            wav_path = random.choice(wav_files)
            azi_deg = random.uniform(0, 360)
            dist_m = random.uniform(0.3, max_distance)
            max_start = max(0, total_samples - int(5 * sr))
            start_sample = random.randint(0, max_start)

            # Retrieve data in main thread
            try:
                dry_mono = _audio_cache[wav_path]
                duration_sec = len(dry_mono) / sr
                
                # Movement logic
                start_azi = random.uniform(0, 360)
                start_dist = random.uniform(0.3, max_distance)
                
                if random.random() < moving_prob and duration_sec > 0.5:
                    # Moving sound
                    velocity = random.uniform(-max_velocity, max_velocity)
                    delta_azi = velocity * duration_sec
                    end_azi = start_azi + delta_azi
                    # Keep distance constant for now (or vary it slightly)
                    end_dist = start_dist 
                else:
                    # Static sound
                    end_azi = start_azi
                    end_dist = start_dist

                # Pack job (removed hrir_L/R from args, worker looks them up)
                spat_jobs.append((i, dry_mono, start_azi, start_dist, end_azi, end_dist, max_distance,
                                  start_sample, _shm.name, buf_shape, random.randint(0, 999999)))
                
                # Store event metadata
                # For moving sounds, we need start/end azi to interpolate labels
                events.append({
                    'start_sample': start_sample,
                    'idx': i,
                    'start_azi': start_azi,
                    'end_azi': end_azi,
                    'start_dist': start_dist,
                    'end_dist': end_dist
                })
            except Exception:
                continue

        if not spat_jobs:
            continue

        # Dispatch batch
        spat_iterator = _persistent_pool.map(_spatialize_sound, spat_jobs, chunksize=10)

        # Collect results
        for i, start_sample, spatial_L, spatial_R in spat_iterator:
            end_sample = min(start_sample + len(spatial_L), total_samples)
            length = end_sample - start_sample
            
            if length > 0:
                stereo_buffer[0, start_sample:end_sample] += spatial_L[:length]
                stereo_buffer[1, start_sample:end_sample] += spatial_R[:length]
                
                # Find the event for this index and update end
                # (Note: events list grows sequentially, so i usually matches index if no skips)
                # But safer to just store them in a dict or update by scanning?
                # Actually, we can just update the correct event in the 'events' list
                # But 'events' list indices don't match 'i' if some skipped.
                # Since we append sequentially in the job loop, the last N events correspond to this batch.
                # Wait, 'i' is the absolute sound index.
                # Let's just trust that we appended an event for every job.
                
                # Store sound data for intensity
                sound_data_list[i] = (spatial_L[:length], spatial_R[:length])
                
                # Update event end sample. We can't easily find the dict in the list by 'i' efficiently.
                # Optim: Store events in a dict temporarily?
                pass
            
            overall_pbar.update(1)

    overall_pbar.close()

    # Re-build events list with correct end samples from sound_data_list
    # Actually, we need 'end' for the events list.
    # Let's fix the events list structure:
    # We only have 'start', 'azi', 'idx' in events.
    # We can infer 'end' from sound_data_list[idx] length.
    
    final_events = []
    for evt in events:
        idx = evt['idx']
        if sound_data_list[idx] is not None:
            s_L, _ = sound_data_list[idx]
            evt['end_sample'] = evt['start_sample'] + len(s_L)
            final_events.append(evt)
            
    events = final_events
    sound_data = sound_data_list

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

    spec_jobs = []
    for win_idx in range(num_windows):
        win_start = win_idx * hop_samples
        win_end = win_start + window_samples
        spec_jobs.append((win_idx, win_start, win_end, _shm.name, buf_shape, sr, n_fft, hop_length, n_gcc_bins))

    chunks = np.zeros((num_windows, NUM_FEATURE_CHANNELS, F_max, T), dtype=np.float32)

    results = tqdm(_persistent_pool.map(_compute_spectrogram, spec_jobs, chunksize=32),
                        total=num_windows, desc="Features", leave=False)

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
        for ev in events:
            ev_start, ev_end = ev['start_sample'], ev['end_sample']
            
            overlap = min(ev_end, win_end) - max(ev_start, snapshot)
            if overlap >= min_overlap_samples:
                s_L, s_R = sound_data[ev['idx']]
                local_start = max(0, snapshot - ev_start)
                local_end = min(len(s_L), win_end - ev_start)
                
                if local_end > local_start:
                    seg = np.concatenate([s_L[local_start:local_end],
                                          s_R[local_start:local_end]])
                    rms = np.sqrt(np.mean(seg**2))
                    
                    # Calculate Azimuth at this specific time window
                    # Linear interpolation based on time progress of the sound
                    # Current time of window center relative to sound start
                    win_center_sample = (win_start + win_end) / 2
                    progress = (win_center_sample - ev_start) / (ev_end - ev_start)
                    progress = np.clip(progress, 0.0, 1.0)
                    
                    current_azi = ev['start_azi'] + (ev['end_azi'] - ev['start_azi']) * progress
                    
                    active.append((current_azi, rms))
                else:
                    pass

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
