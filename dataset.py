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
import HRTF_convolver
from hrir_cache import cache

from convert_wav import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_N_FFT,
    DEFAULT_HOP_LENGTH,
    DEFAULT_WINDOW_SIZE_SECONDS,
    DEFAULT_GCC_MAX_TAU,
    DEFAULT_FREQ_BINS,
    NUM_SPATIAL_CHANNELS,
    NUM_GCC_CHANNELS,
    compute_spatial_features,
)

# Initialize persistent pool globally using all available cores minus one
_persistent_pool = ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 1))
# _persistent_pool = ProcessPoolExecutor(max_workers=8)

MAX_RMS_THRESHOLD = 0.15
RMS_WINDOW_DURATION = 0.05  # 50ms for instantaneous intensity

# Match slab's HRIR samplerate to our audio pipeline
slab.set_default_samplerate(DEFAULT_SAMPLE_RATE)


# ============================================================
# Caches (module-level, persist across epochs)
# ============================================================

_wav_paths_cache = {}  # sounds_dir -> list[str]
_audio_cache = {}  # wav_path -> np.ndarray (samples,)  mono at target sr

# Shared Memory for HRIRs
# (Uses global cache from hrir_cache)


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

    spatial_features, gcc_features = compute_spatial_features(
        ch0,
        ch1,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_gcc_bins=n_gcc_bins,
    )  # (4, F_max, T) and (1, n_gcc_bins, T)

    return win_idx, spatial_features, gcc_features


def _spatialize_sound(args):
    """Worker: convolve mono sound with HRIR and add to shared memory buffer."""
    (
        idx,
        sound_obj,
        start_sample,
        shm_name,
        buf_shape,
        hrir_shm_meta,
        seed,
    ) = args

    # Reseed to ensure different randomness if needed
    np.random.seed(seed)

    # Attach to shared memory for HRIR if not already
    if cache.shm_meta is None and hrir_shm_meta is not None:
        cache.attach(hrir_shm_meta)
    elif cache.shm_meta is None:
        cache.initialize(use_shared_memory=True, quiet=True)

    import HRTF_convolver
    from convert_wav import DEFAULT_SAMPLE_RATE

    stereo_buffer = sound_obj.compute_stereo(normalize=False)
    spatial_L = stereo_buffer[0]
    spatial_R = stereo_buffer[1]

    return idx, start_sample, spatial_L, spatial_R


# ============================================================
# Main generation function
# ============================================================

_initialized = False
_filtered_wav_files = []
_cached_T = None
_stereo_buffer_template = None  # pre-allocated zeros buffer
_shm = None  # persistent shared memory block


def generate_epoch(
    total_duration_seconds=300,
    num_sounds=50,
    window_size_seconds=DEFAULT_WINDOW_SIZE_SECONDS,
    update_interval_ms=100,
    sr=DEFAULT_SAMPLE_RATE,
    n_fft=DEFAULT_N_FFT,
    hop_length=DEFAULT_HOP_LENGTH,
    azi_bins=180,
    room_size=[10.0, 10.0, 3.0],
    listener_pos=[5.0, 5.0, 1.5],
    sigma_deg=13.0,
    sounds_dir=r"sounds\sounds",
    # num_workers=14,
    num_workers=8,
    moving_prob=0.5,  # Probability that a sound moves
    max_speed=5.0,
):  # Max speed in meters/second
    """Generate a fresh random dataset in memory using cached HRIRs and audio.

    Returns:
        chunks_spatial: np.ndarray of shape (num_windows, NUM_SPATIAL_CHANNELS, F_max, T)
        chunks_gcc: np.ndarray of shape (num_windows, NUM_GCC_CHANNELS, n_gcc_bins, T)
        labels: np.ndarray of shape (num_windows, azi_bins)
        metadata_list: list of list of dicts (one list per window, containing metadata for active sounds)
        stereo_buffer: np.ndarray of shape (2, total_samples)
    """
    global _initialized, _filtered_wav_files, _cached_T, _persistent_pool, _shm

    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 1)

    # Calculate dynamic max distance from room boundaries (closest wall)
    lx, ly, _ = listener_pos
    rx, ry, _ = room_size
    max_distance = min(lx, rx - lx, ly, ry - ly)

    # --- One-time initialization ---
    if not _initialized:
        preload_audio(sounds_dir, sr=sr)

        cache.initialize(pool=_persistent_pool, use_shared_memory=True, max_distance=max_distance, room_size=room_size, listener_pos=listener_pos, quiet=False)

        wav_files = _get_wav_paths(sounds_dir)
        _filtered_wav_files = [f for f in wav_files if f in _audio_cache]

        # Precompute T (number of STFT time frames)
        window_samples = int(window_size_seconds * sr)
        dummy_stft = librosa.stft(np.zeros(window_samples, dtype=np.float32), n_fft=n_fft, hop_length=hop_length)
        _cached_T = dummy_stft.shape[1]

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
    events = []  # (start, end, azi_deg)

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
            max_start = max(0, total_samples - int(5 * sr))
            start_sample = random.randint(0, max_start)

            # Retrieve data in main thread
            try:
                dry_mono = _audio_cache[wav_path]

                # Movement logic and object creation
                sound_obj = HRTF_convolver.SpatialSound.generate_random(dry_mono=dry_mono, sr=sr, max_distance=max_distance, moving_prob=moving_prob, max_speed=max_speed)

                # Pack job
                spat_jobs.append(
                    (
                        i,
                        sound_obj,
                        start_sample,
                        _shm.name,
                        buf_shape,
                        cache.shm_meta,
                        random.randint(0, 999999),
                    )
                )

                # Store event metadata
                events.append(
                    {
                        "idx": i,
                        "start_sample": start_sample,
                        "sound_obj": sound_obj,
                    }
                )
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
        idx = evt["idx"]
        if sound_data_list[idx] is not None:
            s_L, _ = sound_data_list[idx]
            evt["end_sample"] = evt["start_sample"] + len(s_L)
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

    chunks_spatial = np.zeros((num_windows, NUM_SPATIAL_CHANNELS, F_max, T), dtype=np.float32)
    chunks_gcc = np.zeros((num_windows, NUM_GCC_CHANNELS, n_gcc_bins, T), dtype=np.float32)

    results = tqdm(_persistent_pool.map(_compute_spectrogram, spec_jobs, chunksize=32), total=num_windows, desc="Features", leave=False)

    for win_idx, spec, gcc in results:
        chunks_spatial[win_idx] = spec
        chunks_gcc[win_idx] = gcc

    # --- 3. Labels with intensity-scaled blobs ---
    rms_window_samples = int(RMS_WINDOW_DURATION * sr)

    labels = np.zeros((num_windows, azi_bins), dtype=np.float32)
    metadata_list = [[] for _ in range(num_windows)]
    # sound_count = np.zeros(num_windows, dtype=np.int32) # No longer needed for filterin

    for win_idx in range(num_windows):
        win_start = win_idx * hop_samples
        win_end = win_start + window_samples

        # New logic: Check for sounds active at the EXACT end of the window
        active_events = []
        min_samples = int(0.070 * sr)  # 70 ms constraint

        for ev in events:
            # Check if the sound covers the instant 'win_end'
            # (start < win_end <= end)
            if ev["start_sample"] < win_end <= ev["end_sample"]:

                # Additional Constraint: Must appear for at least 70ms in this window
                # Since it's active at win_end, the overlap end is win_end.
                # Overlap start is max(win_start, ev['start_sample'])
                overlap_start = max(win_start, ev["start_sample"])
                duration_in_window = win_end - overlap_start

                if duration_in_window >= min_samples:
                    active_events.append(ev)

        if not active_events:
            continue

        for ev in active_events:
            # Calculate instantaneous RMS at win_end
            # Extract segment ending at win_end
            # We need to look up data in sound_data (which has the full spatialized sound)
            s_L, s_R = sound_data[ev["idx"]]

            # Map win_end to local index in the sound array
            local_end_idx = win_end - ev["start_sample"]
            local_start_idx = max(0, local_end_idx - rms_window_samples)

            # If segment is valid
            if local_end_idx > 0:
                seg_L = s_L[local_start_idx:local_end_idx]
                seg_R = s_R[local_start_idx:local_end_idx]

                if len(seg_L) > 0:
                    seg_rms = np.sqrt(np.mean(np.concatenate([seg_L, seg_R]) ** 2))
                    if seg_rms < 1e-4:  # effectively silent
                        continue

                    # Calculate spectral centroid using slab
                    mono_seg = (seg_L + seg_R) * 0.5
                    # try:
                    temp_sound = slab.Sound(data=mono_seg, samplerate=sr)
                    centroid_feat = temp_sound.spectral_feature("centroid")
                    centroid = float(centroid_feat[0] if isinstance(centroid_feat, (list, np.ndarray)) else centroid_feat)
                    # except Exception:
                    #     centroid = 1000.0

                    # Map centroid to a physical radius (meters)
                    # Bass (< 100 Hz) gets a large radius, high (> 4000 Hz) gets a small radius
                    f_min = 100.0
                    f_max = 4000.0
                    centroid_clipped = np.clip(centroid, f_min, f_max)
                    log_f = np.log10(centroid_clipped)
                    log_min = np.log10(f_min)
                    log_max = np.log10(f_max)

                    progress_f = (log_f - log_min) / (log_max - log_min)

                    r_max = 2.5  # Max radius for bass
                    r_min = 0.2  # Min radius for high frequencies
                    radius = r_max - progress_f * (r_max - r_min)

                    # Calculate Azimuth and Distance at win_end using the unified trajectory logic
                    sound_obj = ev["sound_obj"]
                    progress = (win_end - ev["start_sample"]) / (ev["end_sample"] - ev["start_sample"])
                    current_azi, current_dist = sound_obj.get_pos(progress)

                    # Compute angular width based on distance and radius from listener perspective
                    if current_dist <= radius:
                        width_deg = 180.0
                    else:
                        half_angle_rad = np.arcsin(radius / current_dist)
                        width_deg = 2.0 * np.degrees(half_angle_rad)

                    width_bins = max(1, int(round(width_deg / 360 * azi_bins)))

                    azi_idx = round(current_azi / 360 * azi_bins) % azi_bins

                    # Accumulate (flat top + gaussian tails)
                    azi_idx_float = current_azi / 360 * azi_bins
                    width_bins_half = width_deg / 360 * azi_bins / 2.0
                    sigma_bins = max(0.1, sigma_deg / 360 * azi_bins)

                    for idx in range(azi_bins):
                        # Circular discrete distance
                        dist = min(abs(idx - azi_idx_float), azi_bins - abs(idx - azi_idx_float))
                        if dist <= width_bins_half:
                            val = 1.0
                        else:
                            val = np.exp(-((dist - width_bins_half) ** 2) / (2 * sigma_bins**2))
                        labels[win_idx, idx] = max(labels[win_idx, idx], val)

                    metadata_list[win_idx].append(
                        {
                            "id": ev["idx"],
                            "sound_obj": sound_obj,
                            "start_sample": ev["start_sample"],
                            "end_sample": ev["end_sample"],
                            "current_pos": (current_azi, current_dist),
                            "win_range": (win_start, win_end),
                            "radius": radius,
                            "width_deg": width_deg,
                        }
                    )

    # Filter out silent windows (where max label is 0)
    # We only want to train on windows that have at least one active sound
    has_active_sound = labels.max(axis=1) > 0
    chunks_spatial = chunks_spatial[has_active_sound]
    chunks_gcc = chunks_gcc[has_active_sound]
    labels = labels[has_active_sound]

    # Filter metadata list
    metadata_list = [m for i, m in enumerate(metadata_list) if has_active_sound[i]]

    return chunks_spatial, chunks_gcc, labels, metadata_list, stereo_buffer.copy()


# --- CLI: quick test ---
if __name__ == "__main__":
    import soundfile as sf
    from convert_wav import DEFAULT_SAMPLE_RATE

    chunks_spatial, chunks_gcc, labels, meta, stereo_buffer = generate_epoch(
        total_duration_seconds=60,
        num_sounds=30,
        update_interval_ms=2000,
    )
    print(f"Chunks Spatial: {chunks_spatial.shape}")
    print(f"Chunks GCC: {chunks_gcc.shape}")
    print(f"Labels: {labels.shape}")
    print(f"Label range: [{labels.min():.4f}, {labels.max():.4f}]")

    output_wav = "debug_epoch_raw_audio.wav"
    print(f"Saving raw stereo buffer to {output_wav} so you can check if it sounds good...")
    sf.write(output_wav, stereo_buffer.T, DEFAULT_SAMPLE_RATE)
    print("Done!")
