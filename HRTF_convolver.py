"""Implementation of a 3D Audio Panner using Cached HRIRs.

Press 'Play' to start playing the default sound file. Move the Azimuth
and Elevation sliders to position the sound in the 3D space.

You can load your own audio file in File/Load audio file. Also, there
are other sound samples in the folder resources/sound.

*IMPORTANT:* For now, the only working format is a mono WAV file at
44100 Hz sample rate and 16 bit depth.
"""

import os
import wave
import itertools
import pickle

import scipy.io
from scipy.io import wavfile
import scipy.spatial
import pyaudio
import scipy.signal
import librosa
from tqdm import tqdm

import numpy as np
import numpy.linalg

# Values of azimuth and distance used for HRIR interpolation.
AZIMUTH_ANGLES = np.arange(0, 360.5, 0.5)
DISTANCE_STEPS = np.round(np.linspace(0.3, 5.0, 40), 2)


# Convert polar to Cartesian for Delaunay triangulation
# Conventions: x = d * sin(azi_rad), y = d * cos(azi_rad) (Clockwise from North)
def polar_to_cartesian(azi, dist):
    rad = np.radians(azi)
    return dist * np.sin(rad), dist * np.cos(rad)


def cartesian_to_polar(x, y):
    dist = np.sqrt(x**2 + y**2)
    azi = np.degrees(np.arctan2(x, y)) % 360
    return azi, dist


_polar_points = list(itertools.product(AZIMUTH_ANGLES, DISTANCE_STEPS))
POINTS = np.array([polar_to_cartesian(a, d) for a, d in _polar_points])
_POINTS_POLAR = np.array(_polar_points)  # Store to lookup dict quickly

# Initialization variables for audio stream.
SAMPLE_RATE = 44100


def load_and_resample(file_path, target_sr=SAMPLE_RATE):
    """Load an audio file, convert it to mono, and resample if necessary."""
    try:
        import soundfile as sf

        data, sr = sf.read(file_path)
    except Exception as e:
        # Fall back to scipy wavfile if soundfile fails
        sr, data = scipy.io.wavfile.read(file_path)
        data = data.astype(np.float32) / 32768.0  # normalize

    if data.ndim > 1:
        data = data[:, 0]

    if sr != target_sr:
        print(f"Resampling from {sr} to {target_sr}...")
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)

    return data


def butter_lp(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = scipy.signal.butter(N=order, Wn=normal_cutoff, btype="lowpass", output="sos")
    return sos


def butter_lp_filter(signal, cutoff, fs=SAMPLE_RATE, order=1):
    sos = butter_lp(cutoff=cutoff, fs=fs, order=order)
    out = scipy.signal.sosfiltfilt(sos, signal)
    return out


def butter_hp(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = scipy.signal.butter(N=order, Wn=normal_cutoff, btype="highpass", output="sos")
    return sos


def butter_hp_filter(signal, cutoff, fs=SAMPLE_RATE, order=1):
    """Filter a signal with the filter designed in ´butter_hp´.

    Filfilt applies the linear filter twice, once forward and once
    backwards, so that he combined filter has zero phase delay."""

    sos = butter_hp(cutoff=cutoff, fs=fs, order=order)
    out = scipy.signal.sosfiltfilt(sos, signal)
    return out


def create_triangulation(points):
    triangulation = scipy.spatial.Delaunay(points)
    return triangulation


def calculate_T_inv(triang, points):
    """Performs the calculation of the inverse of matrix T for all
    triangles in the triangulation and stores it in an array.

    Matrix T is defined as:

        T = [[A - C],
             [B - C]]

    where A, B and C are vertices of the triangle.

    Since T is independent of source position X, the precalculation of T
    allows to reduce the operational counts for finding the
    interpolation weights.

    For a more comprehensive explanation of this procedure, refer to:

    Gamper, H., Head-related transfer function interpolation in azimuth,
    elevation, and distance, J. Acoust. Soc. Am. 134 (6), December 2013.

    """

    A = points[triang.simplices][:, 0, :]
    B = points[triang.simplices][:, 1, :]
    C = points[triang.simplices][:, 2, :]

    T = np.empty((2 * A.shape[0], A.shape[1]))
    T[::2, :] = A - C
    T[1::2, :] = B - C

    T = T.reshape(-1, 2, 2)
    T_inv = np.linalg.inv(T)
    return T_inv


def interp_hrir(triang, points, T_inv, hrir_dict, azimuth, distance):
    azimuth = float(azimuth) % 360.0
    distance = np.clip(float(distance), DISTANCE_STEPS[0], DISTANCE_STEPS[-1])

    x, y = polar_to_cartesian(azimuth, distance)
    position = [x, y]
    triangle = triang.find_simplex(position)

    if triangle == -1:
        dists = np.sum((points - position) ** 2, axis=1)
        nearest = np.argmin(dists)
        n_azi, n_dist = _POINTS_POLAR[nearest]
        azi_key = float(n_azi) if n_azi < 360.0 else 0.0
        dist_key = round(float(n_dist), 2)
        h_l, h_r = hrir_dict[(azi_key, dist_key)]
        return h_l.copy(), h_r.copy()

    vert = points[triang.simplices[triangle]]
    vert_polar = _POINTS_POLAR[triang.simplices[triangle]]

    X = position - vert[2]
    g = np.dot(X, T_inv[triangle])

    g_1, g_2 = g[0], g[1]
    g_3 = 1 - g_1 - g_2

    if g_1 >= 0 and g_2 >= 0 and g_3 >= 0:
        azi0 = float(vert_polar[0][0]) if vert_polar[0][0] < 360.0 else 0.0
        h0_l, h0_r = hrir_dict[(azi0, round(float(vert_polar[0][1]), 2))]

        azi1 = float(vert_polar[1][0]) if vert_polar[1][0] < 360.0 else 0.0
        h1_l, h1_r = hrir_dict[(azi1, round(float(vert_polar[1][1]), 2))]

        azi2 = float(vert_polar[2][0]) if vert_polar[2][0] < 360.0 else 0.0
        h2_l, h2_r = hrir_dict[(azi2, round(float(vert_polar[2][1]), 2))]

        max_len = max(len(h0_l), len(h1_l), len(h2_l))

        def pad(h, ml):
            if len(h) < ml:
                return np.pad(h, (0, ml - len(h)))
            return h

        h0_l = pad(h0_l, max_len)
        h0_r = pad(h0_r, max_len)
        h1_l = pad(h1_l, max_len)
        h1_r = pad(h1_r, max_len)
        h2_l = pad(h2_l, max_len)
        h2_r = pad(h2_r, max_len)

        return (g_1 * h0_l + g_2 * h1_l + g_3 * h2_l), (g_1 * h0_r + g_2 * h1_r + g_3 * h2_r)

    # Fallback to nearest
    dists = np.sum((vert - position) ** 2, axis=1)
    nearest = np.argmin(dists)
    n_azi, n_dist = vert_polar[nearest]
    azi_key = float(n_azi) if n_azi < 360.0 else 0.0
    dist_key = round(float(n_dist), 2)
    h_l, h_r = hrir_dict[(azi_key, dist_key)]
    return h_l.copy(), h_r.copy()


TRI = create_triangulation(points=POINTS)
T_INV = calculate_T_inv(triang=TRI, points=POINTS)

# Shared Memory for HRIRs
HRIR_CACHE_PATH = "hrir_cache.pkl"
_hrir_shm_meta = None  # dict: (azi, dist) -> (offset_L, offset_R, length) or directly (h_L, h_R) if from disk
_hrir_shm_array = None  # ndarray mapped to shared memory


def load_hrir_cache():
    """Load precomputed HRIRs from grid cache (disk only)."""
    global _hrir_shm_meta, _hrir_shm_array
    local_cache = {}
    if os.path.exists(HRIR_CACHE_PATH):
        print(f"Loading HRIR cache from {HRIR_CACHE_PATH}...")
        with open(HRIR_CACHE_PATH, "rb") as f:
            local_cache = pickle.load(f)
        print(f"Loaded {len(local_cache)} cached HRIRs")
    else:
        print(f"WARNING: {HRIR_CACHE_PATH} not found. Run dataset generation first to create it.")
    _hrir_shm_meta = local_cache
    _hrir_shm_array = None


def _lookup_hrir(azi_deg, dist_m, azi_step=0.5, dist_steps=None, max_distance=None, room_size=[10.0, 10.0, 3.0], listener_pos=[5.0, 5.0, 1.5]):
    global _hrir_shm_meta
    if _hrir_shm_meta is None:
        raise RuntimeError("HRIR cache not initialized.")
    if max_distance is None:
        lx, ly, _ = listener_pos
        rx, ry, _ = room_size
        max_distance = min(lx, rx - lx, ly, ry - ly)
    if dist_steps is None:
        dist_steps = np.linspace(0.3, max_distance, 40)
    azi_snapped = float(round(azi_deg / azi_step) * azi_step % 360)
    dist_snapped = round(float(dist_steps[np.argmin(np.abs(dist_steps - dist_m))]), 2)
    try:
        hrir_L, hrir_R = _hrir_shm_meta[(azi_snapped, dist_snapped)]
    except KeyError:
        return None, None
    return hrir_L, hrir_R


def generate_moving_sound(dry_data, sr):
    import random
    from convert_wav import DEFAULT_SAMPLE_RATE

    n_samples = len(dry_data)
    duration_sec = n_samples / sr

    global _hrir_shm_meta
    if _hrir_shm_meta is None:
        raise RuntimeError("HRIR cache not initialized. Call load_hrir_cache() first.")

    M = max(len(h_L) for h_L, h_R in _hrir_shm_meta.values())
    L = 512  # Window size for the input chunk

    out_length = n_samples + M - 1
    spatial_L = np.zeros(out_length, dtype=np.float32)
    spatial_R = np.zeros(out_length, dtype=np.float32)

    print(f"Generating movement over {duration_sec:.2f} seconds using overlap-add...")

    start_azi = random.uniform(0, 360)
    curr_dist = random.uniform(DISTANCE_STEPS[0], DISTANCE_STEPS[-1])

    cutoff_freq = 200.0
    dry_data_lp = butter_lp_filter(signal=dry_data, cutoff=cutoff_freq)
    dry_data = butter_hp_filter(signal=dry_data, cutoff=cutoff_freq)

    from scipy.signal import fftconvolve, get_window

    hop = L // 2
    window = get_window("hann", L).astype(np.float32)

    for b_start in range(0, n_samples - L + 1, hop):
        b_end = b_start + L
        x_r = dry_data[b_start:b_end]
        x_r_lp = dry_data_lp[b_start:b_end]

        x_r = x_r * window
        x_r_lp = x_r_lp * window

        t_center = (b_start + hop) / sr  # Center of the windowed segment

        curr_azi = (start_azi + t_center * 30.0) % 360
        curr_dist_osc = curr_dist + np.sin(2 * np.pi * t_center / 5.0) * max(0.5, curr_dist * 0.5)
        curr_dist_osc = np.clip(curr_dist_osc, DISTANCE_STEPS[0], DISTANCE_STEPS[-1])

        h_l, h_r = interp_hrir(triang=TRI, points=POINTS, T_inv=T_INV, hrir_dict=_hrir_shm_meta, azimuth=curr_azi, distance=curr_dist_osc)

        conv_L = fftconvolve(x_r, h_l, mode="full")
        conv_R = fftconvolve(x_r, h_r, mode="full")

        # Align low frequencies to the peak of the direct path of the HRIR
        peak_idx = np.argmax(np.abs(h_l))
        lp_start = b_start + peak_idx

        if lp_start < out_length:
            write_len_lp = min(len(x_r_lp), out_length - lp_start)
            spatial_L[lp_start : lp_start + write_len_lp] += x_r_lp[:write_len_lp]
            spatial_R[lp_start : lp_start + write_len_lp] += x_r_lp[:write_len_lp]

        l_conv = len(conv_L)
        target_end = min(b_start + l_conv, out_length)
        write_len_hp = target_end - b_start

        if write_len_hp > 0:
            spatial_L[b_start:target_end] += conv_L[:write_len_hp]
            spatial_R[b_start:target_end] += conv_R[:write_len_hp]

    stereo_buffer = np.stack((spatial_L, spatial_R), axis=0)
    peak = np.max(np.abs(stereo_buffer))
    if peak > 0:
        stereo_buffer = stereo_buffer / peak * 0.9

    return stereo_buffer


if __name__ == "__main__":
    import soundfile as sf
    from convert_wav import DEFAULT_SAMPLE_RATE

    sr = DEFAULT_SAMPLE_RATE

    audio_path = r"mea.wav"
    output_path = "moving_sound.wav"

    print(f"Loading audio file: {audio_path}")
    if not os.path.exists(audio_path):
        print(f"Error: {audio_path} not found.")
    else:
        dry_data = load_and_resample(audio_path, target_sr=sr)

        load_hrir_cache()
        stereo_buffer = generate_moving_sound(dry_data, sr)

        print(f"Writing to {output_path}...")
        sf.write(output_path, stereo_buffer.T, sr)
        print("Done!")
