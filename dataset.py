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
HRIR_CACHE_PATH = "hrir_cache.pkl"
_hrir_shm = None
_hrir_shm_meta = None  # dict: (azi, dist) -> (offset_L, offset_R, length)
_hrir_shm_array = None  # ndarray mapped to shared memory


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


def _compute_hrir_worker(args):
    """Worker: compute HRIR for a specific azimuth and distance."""
    azi_deg, dist_m, room_size, listener_pos = args
    room = slab.Room(size=room_size, listener=listener_pos)
    room.set_source([float(azi_deg), 0, float(dist_m)])
    hrir = room.hrir()
    hrir_data = hrir.data  # (n_taps, 2)
    return float(azi_deg), round(dist_m, 2), hrir_data[:, 0].copy(), hrir_data[:, 1].copy()


def precompute_hrir_grid(azi_step=0.5, dist_steps=None, max_distance=None, room_size=[10.0, 10.0, 3.0], listener_pos=[5.0, 5.0, 1.5]):
    """Precompute HRIRs on a grid and cache them (disk + memory)."""
    global _hrir_shm, _hrir_shm_meta, _hrir_shm_array

    if max_distance is None:
        # Calculate max horizontal distance from listener to the closest wall
        lx, ly, _ = listener_pos
        rx, ry, _ = room_size
        max_distance = min(lx, rx - lx, ly, ry - ly)

    if dist_steps is None:
        dist_steps = np.linspace(0.3, max_distance, 40).tolist()

    # 360 degrees / 0.5 step = 720 azi steps
    num_azi_steps = int(360 / azi_step)
    total = num_azi_steps * len(dist_steps)

    local_cache = {}

    # Try loading from disk first
    if os.path.exists(HRIR_CACHE_PATH):
        print(f"Loading HRIR cache from {HRIR_CACHE_PATH}...")
        with open(HRIR_CACHE_PATH, "rb") as f:
            local_cache = pickle.load(f)
        print(f"Loaded {len(local_cache)} cached HRIRs")

    # Check if we already have everything
    # Need to generate float sequence since range doesn't support floats
    azi_range = np.arange(0, 360, azi_step)
    already_cached = sum(1 for a in azi_range for d in dist_steps if (float(a), round(d, 2)) in local_cache)

    if already_cached < total:
        print(f"Precomputing HRIR grid ({num_azi_steps} azimuths × {len(dist_steps)} distances = {total} positions)...")

        # Identify missing coordinates
        missing_jobs = []
        for azi_deg in azi_range:
            for dist_m in dist_steps:
                key = (float(azi_deg), round(dist_m, 2))
                if key not in local_cache:
                    missing_jobs.append((float(azi_deg), float(dist_m), room_size, listener_pos))

        if missing_jobs:
            global _persistent_pool
            results = tqdm(_persistent_pool.map(_compute_hrir_worker, missing_jobs, chunksize=50), total=len(missing_jobs), desc="Generating HRIRs", leave=False)

            for azi_deg, dist_m, hrir_L, hrir_R in results:
                local_cache[(azi_deg, dist_m)] = (hrir_L, hrir_R)

            # Save to disk
            with open(HRIR_CACHE_PATH, "wb") as f:
                pickle.dump(local_cache, f)
            print(f"HRIR cache: {len(local_cache)} entries (saved to {HRIR_CACHE_PATH})")

    # Move to Shared Memory if not already initialized
    if _hrir_shm is None and local_cache:
        print("Setting up shared memory for HRIR cache...")

        # Determine total size needed
        total_floats = 0
        tap_length = 0
        for k, (hL, hR) in local_cache.items():
            if tap_length == 0:
                tap_length = len(hL)
            total_floats += len(hL) + len(hR)

        nbytes = total_floats * 4  # float32

        # Create shared memory block
        try:
            # Try to attach if it exists from previous crashed run
            _hrir_shm = shared_memory.SharedMemory(name="hrir_cache_shm", create=False)
            _hrir_shm.close()
            _hrir_shm.unlink()
        except FileNotFoundError:
            pass

        _hrir_shm = shared_memory.SharedMemory(name="hrir_cache_shm", create=True, size=nbytes)
        _hrir_shm_array = np.ndarray((total_floats,), dtype=np.float32, buffer=_hrir_shm.buf)
        _hrir_shm_meta = {}

        offset = 0
        for k, (hL, hR) in local_cache.items():
            length = len(hL)

            # Write L
            _hrir_shm_array[offset : offset + length] = hL
            off_L = offset
            offset += length

            # Write R
            _hrir_shm_array[offset : offset + length] = hR
            off_R = offset
            offset += length

            _hrir_shm_meta[k] = (off_L, off_R, length)

        print(f"HRIR Shared Memory initialized. Size: {nbytes / 1024 / 1024:.2f} MB")


def _lookup_hrir(azi_deg, dist_m, azi_step=0.5, dist_steps=None, max_distance=None, room_size=[10.0, 10.0, 3.0], listener_pos=[5.0, 5.0, 1.5], shm_meta=None):
    """Find nearest cached HRIR for given azimuth and distance."""
    global _hrir_shm, _hrir_shm_array, _hrir_shm_meta

    # Worker initialization of shared memory wrapper
    if _hrir_shm_meta is None and shm_meta is not None:
        _hrir_shm_meta = shm_meta
        _hrir_shm = shared_memory.SharedMemory(name="hrir_cache_shm", create=False)

        # Calculate size based on first item
        total_floats = 0
        for k, (off_L, off_R, length) in _hrir_shm_meta.items():
            total_floats += 2 * length
        _hrir_shm_array = np.ndarray((total_floats,), dtype=np.float32, buffer=_hrir_shm.buf)

    if _hrir_shm_meta is None:
        raise RuntimeError("HRIR shared memory metadata not found in worker process.")

    if max_distance is None:
        lx, ly, _ = listener_pos
        rx, ry, _ = room_size
        max_distance = min(lx, rx - lx, ly, ry - ly)

    if dist_steps is None:
        dist_steps = np.linspace(0.3, max_distance, 40)

    # Snap azimuth to nearest grid point
    azi_snapped = float(round(azi_deg / azi_step) * azi_step % 360)

    # Snap distance to nearest grid point
    dist_snapped = round(float(dist_steps[np.argmin(np.abs(dist_steps - dist_m))]), 2)

    off_L, off_R, length = _hrir_shm_meta[(azi_snapped, dist_snapped)]
    hrir_L = _hrir_shm_array[off_L : off_L + length]
    hrir_R = _hrir_shm_array[off_R : off_R + length]

    return hrir_L, hrir_R


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
    (idx, wav_path, start_azi, start_dist, end_azi, end_dist, max_distance, room_size, listener_pos, start_sample, shm_name, buf_shape, seed, hrir_shm_meta) = args

    # Reseed to ensure different randomness if needed (though we pass explicit params here)
    np.random.seed(seed)

    dry_mono = wav_path  # args can contain the data directly

    # Initialize shared memory for HRIR if not already
    global _hrir_shm, _hrir_shm_array, _hrir_shm_meta

    if _hrir_shm_meta is None and hrir_shm_meta is not None:
        _hrir_shm_meta = hrir_shm_meta
        _hrir_shm = shared_memory.SharedMemory(name="hrir_cache_shm", create=False)
        total_floats = 0
        for k, (off_L, off_R, length) in _hrir_shm_meta.items():
            total_floats += 2 * length
        _hrir_shm_array = np.ndarray((total_floats,), dtype=np.float32, buffer=_hrir_shm.buf)

    import HRTF_convolver
    from convert_wav import DEFAULT_SAMPLE_RATE

    class HRIRDictWrapper:
        def __init__(self, meta, array):
            self.meta = meta
            self.array = array

        def __getitem__(self, key):
            off_L, off_R, length = self.meta[key]
            return self.array[off_L : off_L + length].copy(), self.array[off_R : off_R + length].copy()

        def values(self):
            for off_L, off_R, length in self.meta.values():
                yield self.array[off_L : off_L + length], self.array[off_R : off_R + length]

    wrapper = HRIRDictWrapper(_hrir_shm_meta, _hrir_shm_array)

    sr = DEFAULT_SAMPLE_RATE

    stereo_buffer = HRTF_convolver.generate_moving_sound(
        dry_data=dry_mono, sr=sr, start_azi=start_azi, start_dist=start_dist, end_azi=end_azi, end_dist=end_dist, hrir_cache=wrapper, normalize=False
    )
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
    max_velocity=90.0,
):  # Max velocity in degrees/second
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
        precompute_hrir_grid(max_distance=max_distance, room_size=room_size, listener_pos=listener_pos)

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
                duration_sec = len(dry_mono) / sr

                # Movement logic
                # Movement logic: Vector-based
                start_azi = random.uniform(0, 360)
                # Distance distribution: Mostly 2-4m, few near 0.
                # Triangular distribution peaking at 3.5m (assuming max~5)
                start_dist = random.triangular(0.5, max_distance, 3.5)

                # Convert to Cartesian
                sa_rad = np.radians(start_azi)
                sx = start_dist * np.sin(sa_rad)
                sy = start_dist * np.cos(sa_rad)

                end_azi = start_azi
                end_dist = start_dist

                if random.random() < moving_prob and duration_sec > 0.5:
                    # Random velocity vector
                    move_heading = random.uniform(0, 360)
                    speed = random.uniform(0.5, 3.0)  # m/s (Reasonable walking/running speed)

                    # Velocity components
                    mh_rad = np.radians(move_heading)
                    vx = speed * np.sin(mh_rad)
                    vy = speed * np.cos(mh_rad)

                    # Unclamped end pos
                    ex = sx + vx * duration_sec
                    ey = sy + vy * duration_sec

                    # Clamp to max_distance
                    # Ray-circle intersection? Or just scale vector?
                    # Since we start inside, we just need to check if we exit.
                    final_dist = np.sqrt(ex**2 + ey**2)

                    if final_dist > max_distance:
                        # We went out of bounds. Find intersection point.
                        # We want t such that |start + v*t| = max_dist
                        # This works, but easier approximation: just clip the end point to boundary?
                        # No, clipping dist changes direction.
                        # We should shorten the path so it stops AT the boundary (or bounces, but let's just stop/slide).
                        # Let's simple clip the distance of the end point? No that warps the path.
                        # Let's scale the vector (ex, ey) to be on the circle? No.
                        # Correct way: find intersection of line segment with circle.
                        # But lazy way (good enough): normalize (ex, ey) to max_dist.
                        # This effectively makes the sound "slide" along the wall if it was going straight out,
                        # but changes trajectory if it was tangential.
                        #
                        # BETTER: Scale down the displacement so it ends at boundary?
                        # No, that changes speed.
                        #
                        # SIMPLEST ROBUST: If end is OOB, retry with new randoms (rejection sampling).
                        # But that might bias towards center.
                        #
                        # Let's just clip the end point to max radius direction-wise from center.
                        # This is what "scale to max_dist" means.
                        # It changes the physical path (curves it to center), but ensures validity.
                        scale = max_distance / final_dist
                        ex *= scale
                        ey *= scale
                        final_dist = max_distance

                    # Convert end back to polar
                    end_dist = final_dist
                    end_azi = np.degrees(np.arctan2(ex, ey)) % 360
                else:
                    # Static
                    end_azi = start_azi
                    end_dist = start_dist

                # Pack job (removed hrir_L/R from args, worker looks them up)
                spat_jobs.append(
                    (
                        i,
                        dry_mono,
                        start_azi,
                        start_dist,
                        end_azi,
                        end_dist,
                        max_distance,
                        room_size,
                        listener_pos,
                        start_sample,
                        _shm.name,
                        buf_shape,
                        random.randint(0, 999999),
                        _hrir_shm_meta,
                    )
                )

                # Store event metadata
                # For moving sounds, we need start/end azi to interpolate labels
                events.append({"start_sample": start_sample, "idx": i, "start_azi": start_azi, "end_azi": end_azi, "start_dist": start_dist, "end_dist": end_dist})
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

                    # Calculate Azimuth and Distance at win_end
                    progress = (win_end - ev["start_sample"]) / (ev["end_sample"] - ev["start_sample"])
                    progress = np.clip(progress, 0.0, 1.0)

                    # Use Cartesian interpolation to match spatialization logic
                    sa_rad = np.radians(ev["start_azi"])
                    ea_rad = np.radians(ev["end_azi"])
                    sx, sy = ev["start_dist"] * np.sin(sa_rad), ev["start_dist"] * np.cos(sa_rad)
                    ex, ey = ev["end_dist"] * np.sin(ea_rad), ev["end_dist"] * np.cos(ea_rad)

                    cur_x = sx + (ex - sx) * progress
                    cur_y = sy + (ey - sy) * progress

                    current_dist = np.sqrt(cur_x**2 + cur_y**2)
                    current_azi = np.degrees(np.arctan2(cur_x, cur_y)) % 360

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

                    # Store metadata for visualization
                    # We want to know: ID, Trajectory (Start->End), Current Pos, Timing, Radius, Width
                    metadata_list[win_idx].append(
                        {
                            "id": ev["idx"],
                            "start_sample": ev["start_sample"],
                            "end_sample": ev["end_sample"],
                            "traj_start": (ev["start_azi"], ev["start_dist"]),
                            "traj_end": (ev["end_azi"], ev["end_dist"]),
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
