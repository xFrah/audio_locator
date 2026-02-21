"""Implementation of a 3D Audio Panner using the CIPIC HRTF Database.

Usage:

Select a subject from the CIPIC database. You should select a subject
with similar anthropometric measurements as yourself for the best
experience.

    - Note: Due to storage limitations, the repository has only 4
            subjects of the database to choose from. The full database
            is ~170MB and has 45 subjects. It can be downloaded for free
            at:

            https://www.ece.ucdavis.edu/cipic/spatial-sound/hrtf-data/

            In order to make it work, you should simply replace the
            folder ´CIPIC_hrtf_database´ with the one you downloaded.

Press 'Play' to start playing the default sound file. Move the Azimuth
and Elevation sliders to position the sound in the 3D space.

You can load your own audio file in File/Load audio file. Also, there
are other sound samples in the folder resources/sound.

*IMPORTANT:* For now, the only working format is a mono WAV file at
44100 Hz sample rate and 16 bit depth.

You can save the file at the specified pair of Azimuth/Elevation in
File/Save audio file.

Lastly, you can choose to use a crossover in order not to spatialize low
frequencies, since low frequencies are non-directional in nature. Go to
Settings/Change cutoff frequency to set the desired frequency. By
default, crossover is set at 200 Hz.

Author:         Francisco Rotea
                (Buenos Aires, Argentina)
Repository:     https://github.com/franciscorotea
Email:          francisco.rotea@gmail.com

"""

import os
import wave
import itertools

import scipy.io
from scipy.io import wavfile
import scipy.spatial
import pyaudio
import scipy.signal
import librosa
import pickle
import slab
from tqdm import tqdm

import numpy as np
import numpy.linalg

# Values of azimuth and elevation angles measured in the CIPIC database.
# See ´CIPIC_hrtf_database/doc/hrir_data_documentation.pdf´ for
# information about the coordinate system and measurement procedure.

AZIMUTH_ANGLES = [
    -80,
    -65,
    -55,
    -45,
    -40,
    -35,
    -30,
    -25,
    -20,
    -15,
    -10,
    -5,
    0,
    5,
    10,
    15,
    20,
    25,
    30,
    35,
    40,
    45,
    55,
    65,
    80,
]

ELEVATION_ANGLES = -45 + 5.625 * np.arange(0, 50)

POINTS = np.array(list(itertools.product(AZIMUTH_ANGLES, ELEVATION_ANGLES)))

# Get indexes from angles.

AZ = dict(zip(AZIMUTH_ANGLES, np.arange(len(AZIMUTH_ANGLES))))
EL = dict(zip(ELEVATION_ANGLES, np.arange(len(ELEVATION_ANGLES))))

# Load anthropometric measurements data from the CIPIC database.

# See ´CIPIC_hrtf_database/doc/anthropometry.pdf´ for information about
# the parameters definition.

anthro_data = scipy.io.loadmat("CIPIC_hrtf_database/anthropometry/anthro.mat")

PARAMETERS = {
    "info": ["Age:", "Sex:", "Weight:"],
    "X": [
        "Head width:",
        "Head height:",
        "Head depth:",
        "Pinna offset down:",
        "Pinna offset back:",
        "Neck width:",
        "Neck height:",
        "Neck depth:",
        "Torso top width:",
        "Torso top height:",
        "Torso top depth:",
        "Shoulder width:",
        "Head offset forward:",
        "Height:",
        "Seated height:",
        "Head circumference:",
        "Shoulder circumference:",
    ],
    "D": ["Cavum concha height", "Cymba concha height", "Cavum concha width", "Fossa height", "Pinna height", "Pinna width", "Intertragal incisure width", "Cavum concha depth"],
    "theta": ["Pinna rotation angle:", "Pinna flare angle:"],
}

L_R = [" (left):", " (right):"]  # To use with 'D' and 'theta' parameters.

# Clean anthropometric data for display.

for key, value in anthro_data.items():
    if key not in ["__header__", "__version__", "__globals__", "id", "sex"]:
        if key == "age":
            anthro_data[key][np.isnan(anthro_data[key])] = 0
            anthro_data[key] = np.squeeze(value.astype("int")).astype("str")
            anthro_data[key][anthro_data[key] == "0"] = "-"
        else:
            anthro_data[key] = np.around(np.squeeze(value), 1).astype("str")
            anthro_data[key][anthro_data[key] == "nan"] = "-"

# Get indexes from ID's.

ANTHRO_ID = anthro_data["id"].flatten().tolist()
ID_TO_IDX = dict(zip(ANTHRO_ID, range(len(ANTHRO_ID))))

# Generate a list with all subject's ID present in the database.

FOLDERS = os.listdir("CIPIC_hrtf_database/standard_hrir_database")
SUBJECT_ID = [id_.strip("subject_") for id_ in FOLDERS if id_ != "show_data"]

# Initialization variables for audio stream.

SAMPLE_RATE = 44100

# Initialization variables for overlap-save algorithm.

# L = Window size.
# M = Length of impulse response.
# N = Size of the DFT. Since the length of the convolved signal will be
#     L+M-1, it is rounded to the nearest power of 2 for efficient fft
#     calculation.

L = 2048
M = 200
N = int(2 ** np.ceil(np.log2(np.abs(L + M - 1))))

L = N - M + 1

# Preallocate interpolated impulse responses.

interp_hrir_l = np.zeros(M)
interp_hrir_r = np.zeros(M)


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
    """Design of a digital Butterworth low pass filter with a
    second-order section format for numerical stability."""

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq

    sos = scipy.signal.butter(N=order, Wn=normal_cutoff, btype="lowpass", output="sos")

    return sos


def butter_lp_filter(signal, cutoff, fs=SAMPLE_RATE, order=1):
    """Filter a signal with the filter designed in ´butter_lp´.

    Filfilt applies the linear filter twice, once forward and once
    backwards, so that he combined filter has zero phase delay."""

    sos = butter_lp(cutoff=cutoff, fs=fs, order=order)
    out = scipy.signal.sosfiltfilt(sos, signal)

    return out


def butter_hp(cutoff, fs, order):
    """Design of a digital Butterworth high pass filter with a
    second-order section format for numerical stability."""

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
    """Generate a triangular mesh from HRTF measurement points (azimuth,
    elevation) using the Delaunay triangulation algorithm."""

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


def interp_hrir(triang, points, T_inv, hrir_l, hrir_r, azimuth, elevation):
    """Estimate a HRTF for any point X lying inside the triangular mesh
    calculated.

    This is done by interpolating the vertices of the triangle enclosing
    X. Given a triangle with vertices A, B and C, any point X inside the
    triangle can be represented as a linear combination of the vertices:

    X = g_1 * A + g_2 * B + g_3 * C

    where g_i are scalar weights. If the sum of the weights is equal to
    1, these are barycentric coordinates of point X. Given a desired
    source position X, barycentric interpolation weights are calculated
    as:

    [g_1, g_2] = (X - C) * T_inv
    g_ 3 = 1 - g_1 - g_2

    Barycentric coordinates are used as interpolation weights for
    estimating the HRTF at point X as the weighted sum of the HRTFs
    measured at A, B and C, respectively.

    One of the main advantages of this interpolation approach is that
    it does not cause discontinuities in the interpolated HRTFs: for a
    source moving smoothly from one triangle to another, the HRTF
    estimate changes smoothly, even at the crossing point.

    For a more comprehensive explanation of the interpolation algorithm,
    please refer to:

    Gamper, H., Head-related transfer function interpolation in azimuth,
    elevation, and distance, J. Acoust. Soc. Am. 134 (6), December 2013.

    """

    position = [azimuth, elevation]
    triangle = triang.find_simplex(position)

    if triangle == -1:
        # Prevent hard crash/pop if point is outside the HRTF convex hull
        dists = np.sum((points - position) ** 2, axis=1)
        nearest = np.argmin(dists)
        interp_hrir_l[:] = hrir_l[AZ[points[nearest][0]]][EL[points[nearest][1]]][:]
        interp_hrir_r[:] = hrir_r[AZ[points[nearest][0]]][EL[points[nearest][1]]][:]
        return interp_hrir_l, interp_hrir_r

    vert = points[triang.simplices[triangle]]

    X = position - vert[2]
    g = np.dot(X, T_inv[triangle])

    g_1 = g[0]
    g_2 = g[1]
    g_3 = 1 - g_1 - g_2

    if g_1 >= 0 and g_2 >= 0 and g_3 >= 0:
        interp_hrir_l[:] = g_1 * hrir_l[AZ[vert[0][0]]][EL[vert[0][1]]][:] + g_2 * hrir_l[AZ[vert[1][0]]][EL[vert[1][1]]][:] + g_3 * hrir_l[AZ[vert[2][0]]][EL[vert[2][1]]][:]

        interp_hrir_r[:] = g_1 * hrir_r[AZ[vert[0][0]]][EL[vert[0][1]]][:] + g_2 * hrir_r[AZ[vert[1][0]]][EL[vert[1][1]]][:] + g_3 * hrir_r[AZ[vert[2][0]]][EL[vert[2][1]]][:]

    return interp_hrir_l, interp_hrir_r


TRI = create_triangulation(points=POINTS)
T_INV = calculate_T_inv(triang=TRI, points=POINTS)

# Shared Memory for HRIRs
HRIR_CACHE_PATH = "hrir_cache.pkl"
_hrir_shm_meta = None  # dict: (azi, dist) -> (offset_L, offset_R, length)
_hrir_shm_array = None  # ndarray mapped to shared memory


def load_hrir_cache():
    """Load precomputed HRIRs from grid cache (disk only)."""
    global _hrir_shm_meta, _hrir_shm_array

    local_cache = {}

    # Try loading from disk first
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
        raise RuntimeError("HRIR cache not initialized. Run precompute_hrir_grid() first.")

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


def _get_interpolated_hrir(azi_deg, dist_m, max_distance=None, room_size=[10.0, 10.0, 3.0], listener_pos=[5.0, 5.0, 1.5]):
    return _lookup_hrir(azi_deg, dist_m, max_distance=max_distance, room_size=room_size, listener_pos=listener_pos)


def generate_moving_sound(dry_data, sr):
    import random
    from convert_wav import DEFAULT_SAMPLE_RATE

    n_samples = len(dry_data)
    duration_sec = n_samples / sr

    print("Loading HRIR data from CIPIC subject 003...")
    subject = "003"
    hrir_mat = scipy.io.loadmat("CIPIC_hrtf_database/standard_hrir_database/subject_" + subject + "/hrir_final.mat")
    hrir_l = np.array(hrir_mat["hrir_l"])
    hrir_r = np.array(hrir_mat["hrir_r"])

    # Overlap-save algorithm parameters from HRTF_convolver.py
    L = 512  # Window size for the input chunk
    M = 200  # HRIR length
    N = int(2 ** np.ceil(np.log2(np.abs(L + M - 1))))  # DFT size

    # Adjusted L so that N is a power of 2
    L = N - M + 1

    buffer_OLAP_L = np.zeros(M - 1)
    buffer_OLAP_R = np.zeros(M - 1)

    # We will accumulate output here. Since output is same length as input + reverb tail (which we can truncate)
    out_length = n_samples
    spatial_L = np.zeros(out_length, dtype=np.float32)
    spatial_R = np.zeros(out_length, dtype=np.float32)

    print(f"Generating movement over {duration_sec} seconds using overlap-save...")

    start_azi = random.uniform(min(AZIMUTH_ANGLES), max(AZIMUTH_ANGLES))
    # Keep elevation near 0
    curr_el = 0.0

    # Filter the entire audio track offline to prevent zero-phase chunking edge artifacts
    cutoff_freq = 200.0
    dry_data_lp = butter_lp_filter(signal=dry_data, cutoff=cutoff_freq)
    dry_data = butter_hp_filter(signal=dry_data, cutoff=cutoff_freq)

    for b_start in range(0, n_samples, L):
        b_end = min(b_start + L, n_samples)
        x_r = dry_data[b_start:b_end]
        x_r_lp = dry_data_lp[b_start:b_end]

        # If last block is shorter than L, pad it with zeros
        padded_chunk = False
        if len(x_r) < L:
            x_r = np.pad(x_r, (0, L - len(x_r)))
            x_r_lp = np.pad(x_r_lp, (0, L - len(x_r_lp)))
            padded_chunk = True

        # Calculate current position
        t_center = (b_start + (b_start + L)) / 2 / sr

        # Move back and forth in azimuth (-80 to 80)
        # Period of 12 seconds
        # Sine wave mapping to azimuth
        curr_azi = np.sin(2 * np.pi * t_center / 12.0) * max(AZIMUTH_ANGLES)

        # Interpolate to get the HRIR at the position selected
        h_l, h_r = interp_hrir(triang=TRI, points=POINTS, T_inv=T_INV, hrir_l=hrir_l, hrir_r=hrir_r, azimuth=curr_azi, elevation=curr_el)

        # Compute DFT of impulse response
        h = np.vstack(([h_l, h_r]))
        h = np.hstack((h, np.zeros((2, N - (M - 1)))))
        H = np.fft.fft(h, N)

        # Overlap Save Algorithm on high frequencies
        x_L_overlap = np.hstack((buffer_OLAP_L, x_r))
        x_L_zeropad = np.hstack((x_L_overlap, np.zeros(N - len(x_L_overlap))))

        x_R_overlap = np.hstack((buffer_OLAP_R, x_r))
        x_R_zeropad = np.hstack((x_R_overlap, np.zeros(N - len(x_R_overlap))))

        # Save overlap for next iteration
        buffer_OLAP_L[:] = x_L_zeropad[N - (M - 1) : N]
        buffer_OLAP_R[:] = x_R_zeropad[N - (M - 1) : N]

        # Convolution using high frequencies only
        Xm_L = np.fft.fft(x_L_zeropad, N)
        Xm_R = np.fft.fft(x_R_zeropad, N)

        Ym_L = Xm_L * H[0]
        Ym_R = Xm_R * H[1]

        ym_L = np.real(np.fft.ifft(Ym_L))
        ym_R = np.real(np.fft.ifft(Ym_R))

        # Add back omnidirectional low frequencies (direct time-aligned addition)
        l_out = ym_L[M - 1 : N + 1] + x_r_lp  # First M-1 samples are Aliased/Discarded
        r_out = ym_R[M - 1 : N + 1] + x_r_lp

        # Determine how much to write
        write_len = min(L, n_samples - b_start)

        if write_len > 0:
            spatial_L[b_start : b_start + write_len] = l_out[:write_len]
            spatial_R[b_start : b_start + write_len] = r_out[:write_len]

    # Normalize
    stereo_buffer = np.stack((spatial_L, spatial_R), axis=0)  # (2, len)
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
