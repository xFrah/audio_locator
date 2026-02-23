"""Implementation of a 3D Audio Panner using Cached HRIRs.

Press 'Play' to start playing the default sound file. Move the Azimuth
and Elevation sliders to position the sound in the 3D space.

You can load your own audio file in File/Load audio file. Also, there
are other sound samples in the folder resources/sound.

*IMPORTANT:* For now, the only working format is a mono WAV file at
44100 Hz sample rate and 16 bit depth.
"""

import os
import itertools
from hrir_cache import cache

import scipy.io
import scipy.spatial
import scipy.signal
import librosa

import numpy as np
from config import *

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



def load_and_resample(file_path):
    """Load an audio file, convert it to mono, and resample if necessary."""
    try:
        import soundfile as sf

        data, sr = sf.read(file_path)
    except Exception as e:
        # Fall back to scipy wavfile if soundfile fails
        import scipy.io.wavfile

        sr, data = scipy.io.wavfile.read(file_path)
        data = data.astype(np.float32) / 32768.0  # normalize

    if data.ndim > 1:
        data = data[:, 0]

    if sr != DEFAULT_SAMPLE_RATE:
        print(f"Resampling from {sr} to {DEFAULT_SAMPLE_RATE}...")
        data = librosa.resample(data, orig_sr=sr, target_sr=DEFAULT_SAMPLE_RATE)

    return data


def butter_lp(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = scipy.signal.butter(N=order, Wn=normal_cutoff, btype="lowpass", output="sos")
    return sos


def butter_lp_filter(signal, cutoff, fs=DEFAULT_SAMPLE_RATE, order=1):
    sos = butter_lp(cutoff=cutoff, fs=fs, order=order)
    out = scipy.signal.sosfiltfilt(sos, signal)
    return out


def butter_hp(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = scipy.signal.butter(N=order, Wn=normal_cutoff, btype="highpass", output="sos")
    return sos


def butter_hp_filter(signal, cutoff, fs=DEFAULT_SAMPLE_RATE, order=1):
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


def interp_hrir(triang, points, T_inv, azimuth, distance):
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
        h_l, h_r = cache[(azi_key, dist_key)]
        return h_l.copy(), h_r.copy()

    vert = points[triang.simplices[triangle]]
    vert_polar = _POINTS_POLAR[triang.simplices[triangle]]

    X = position - vert[2]
    g = np.dot(X, T_inv[triangle])

    g_1, g_2 = g[0], g[1]
    g_3 = 1 - g_1 - g_2

    if g_1 >= 0 and g_2 >= 0 and g_3 >= 0:
        azi0 = float(vert_polar[0][0]) if vert_polar[0][0] < 360.0 else 0.0
        h0_l, h0_r = cache[(azi0, round(float(vert_polar[0][1]), 2))]

        azi1 = float(vert_polar[1][0]) if vert_polar[1][0] < 360.0 else 0.0
        h1_l, h1_r = cache[(azi1, round(float(vert_polar[1][1]), 2))]

        azi2 = float(vert_polar[2][0]) if vert_polar[2][0] < 360.0 else 0.0
        h2_l, h2_r = cache[(azi2, round(float(vert_polar[2][1]), 2))]

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

    raise ValueError("Could not find a valid triangle for the given position.")
    # Fallback to nearest
    dists = np.sum((vert - position) ** 2, axis=1)
    nearest = np.argmin(dists)
    n_azi, n_dist = vert_polar[nearest]
    azi_key = float(n_azi) if n_azi < 360.0 else 0.0
    dist_key = round(float(n_dist), 2)
    h_l, h_r = cache[(azi_key, dist_key)]
    return h_l.copy(), h_r.copy()


TRI = create_triangulation(points=POINTS)
T_INV = calculate_T_inv(triang=TRI, points=POINTS)


class SpatialSound:
    """Encapsulates a mono sound with spatial trajectory properties."""

    def __init__(self, dry_mono, start_dist, start_azi, end_dist, end_azi, is_circular=False, speed=None):
        self.dry_mono = dry_mono
        self.sr = DEFAULT_SAMPLE_RATE
        self.start_dist = start_dist
        self.start_azi = start_azi
        self.end_dist = end_dist
        self.end_azi = end_azi
        self.is_circular = is_circular
        self.speed = speed
        self.is_stationary = (abs(start_azi - end_azi) < 0.5) and (abs(start_dist - end_dist) < 0.1)

    @staticmethod
    def generate_random(dry_mono, max_distance, moving_prob=0.5, max_speed=3.0):
        """Factory method to create a SpatialSound with a random trajectory."""
        sr = DEFAULT_SAMPLE_RATE
        import random

        duration_sec = len(dry_mono) / sr
        start_azi = random.uniform(0, 360)
        # Distance distribution: Mostly 2-4m, few near 0.
        # Triangular distribution peaking at 3.5m (assuming max~5)
        start_dist = random.triangular(0.5, max_distance, 3.5)

        # Convert to Cartesian for vector math
        sa_rad = np.radians(start_azi)
        sx = start_dist * np.sin(sa_rad)
        sy = start_dist * np.cos(sa_rad)

        if random.random() < moving_prob and duration_sec > 0.5:
            is_circular = random.random() < 0.5
            speed = random.uniform(0.5, max_speed)  # m/s

            if not is_circular:
                # Random velocity vector (Linear)
                move_heading = random.uniform(0, 360)
                mh_rad = np.radians(move_heading)
                vx = speed * np.sin(mh_rad)
                vy = speed * np.cos(mh_rad)

                # Unclamped end pos
                ex = sx + vx * duration_sec
                ey = sy + vy * duration_sec

                final_dist = np.sqrt(ex**2 + ey**2)
                if final_dist > max_distance:
                    scale = max_distance / final_dist
                    ex *= scale
                    ey *= scale
                    final_dist = max_distance

                # Convert end back to polar
                end_dist = final_dist
                end_azi = np.degrees(np.arctan2(ex, ey)) % 360
            else:
                # Circular motion
                direction = random.choice([-1.0, 1.0])
                # Just a hint for rotation direction
                end_azi = start_azi + direction * 10
                end_dist = start_dist  # Keep constant distance for circular
        else:
            # Static
            is_circular = False
            speed = None
            end_azi = start_azi
            end_dist = start_dist

        return SpatialSound(
            dry_mono=dry_mono,
            start_dist=start_dist,
            start_azi=start_azi,
            end_dist=end_dist,
            end_azi=end_azi,
            is_circular=is_circular,
            speed=speed,
        )

    def compute_stereo(self, normalize=True):
        """Computes the stereo audio based on the trajectory."""
        cache.initialize()

        dry_data = self.dry_mono
        sr = DEFAULT_SAMPLE_RATE
        n_samples = len(dry_data)

        # Initialize M (tap length) from any HRIR in the cache
        first_hrir_pair = next(iter(cache.values()))
        M = len(first_hrir_pair[0])

        if self.is_stationary:
            from scipy.signal import fftconvolve

            h_l, h_r = interp_hrir(triang=TRI, points=POINTS, T_inv=T_INV, azimuth=self.start_azi, distance=self.start_dist)
            spatial_L = fftconvolve(dry_data, h_l, mode="full").astype(np.float32)
            spatial_R = fftconvolve(dry_data, h_r, mode="full").astype(np.float32)
            stereo_buffer = np.stack((spatial_L, spatial_R), axis=0)

            if normalize:
                peak = np.max(np.abs(stereo_buffer))
                if peak > 0:
                    stereo_buffer = stereo_buffer / peak * 0.9

            return stereo_buffer

        out_length = n_samples + M - 1
        spatial_L = np.zeros(out_length, dtype=np.float32)
        spatial_R = np.zeros(out_length, dtype=np.float32)

        cutoff_freq = 200.0
        dry_data_lp = butter_lp_filter(signal=dry_data, cutoff=cutoff_freq)
        dry_data_hp = butter_hp_filter(signal=dry_data, cutoff=cutoff_freq)

        from scipy.signal import fftconvolve, get_window

        window_size = 2048
        hop = window_size // 2
        window = get_window("hann", window_size).astype(np.float32)

        for b_start in range(0, n_samples - window_size + 1, hop):
            # Use high-pass for convolution
            x_r_hp = dry_data_hp[b_start : b_start + window_size] * window
            x_r_lp = dry_data_lp[b_start : b_start + window_size] * window

            progress = (b_start + hop) / n_samples
            azi1, dist1 = self.get_pos(progress)

            h_l_next, h_r_next = interp_hrir(triang=TRI, points=POINTS, T_inv=T_INV, azimuth=azi1, distance=dist1)

            conv_L_cross = fftconvolve(x_r_hp, h_l_next, mode="full")
            conv_R_cross = fftconvolve(x_r_hp, h_r_next, mode="full")

            # Align low frequencies to the peak of the direct path of the HRIR
            peak_idx = np.argmax(np.abs(h_l_next))
            lp_start = b_start + peak_idx

            if lp_start < out_length:
                write_len_lp = min(len(x_r_lp), out_length - lp_start)
                spatial_L[lp_start : lp_start + write_len_lp] += x_r_lp[:write_len_lp]
                spatial_R[lp_start : lp_start + write_len_lp] += x_r_lp[:write_len_lp]

            l_conv = len(conv_L_cross)
            target_end = min(b_start + l_conv, out_length)
            write_len_hp = target_end - b_start

            if write_len_hp > 0:
                spatial_L[b_start:target_end] += conv_L_cross[:write_len_hp]
                spatial_R[b_start:target_end] += conv_R_cross[:write_len_hp]

        stereo_buffer = np.stack((spatial_L, spatial_R), axis=0)

        if normalize:
            peak = np.max(np.abs(stereo_buffer))
            if peak > 0:
                stereo_buffer = stereo_buffer / peak * 0.9

        return stereo_buffer

    def get_pos(self, progress):
        """Calculates the (azimuth, distance) at a given progress [0, 1]."""
        global DISTANCE_STEPS

        progress = np.clip(progress, 0.0, 1.0)
        duration_sec = len(self.dry_mono) / self.sr
        t = progress * duration_sec

        # Convert start/end to Cartesian for linear interpolation
        sa_rad = np.radians(self.start_azi)
        ea_rad = np.radians(self.end_azi)
        sx, sy = self.start_dist * np.sin(sa_rad), self.start_dist * np.cos(sa_rad)
        ex, ey = self.end_dist * np.sin(ea_rad), self.end_dist * np.cos(ea_rad)

        if self.speed is not None:
            dist_moved = self.speed * t
            if not self.is_circular:
                # Move along the vector from start to end
                dx, dy = ex - sx, ey - sy
                total_path_dist = np.sqrt(dx**2 + dy**2)
                if total_path_dist > 1e-6:
                    cur_x = sx + (dx / total_path_dist) * dist_moved
                    cur_y = sy + (dy / total_path_dist) * dist_moved
                else:
                    cur_x, cur_y = sx, sy

                dist = np.sqrt(cur_x**2 + cur_y**2)
                azi = np.degrees(np.arctan2(cur_x, cur_y)) % 360
            else:
                # Move along the arc
                dist = self.start_dist + (self.end_dist - self.start_dist) * progress
                angle_diff = self.end_azi - self.start_azi
                if abs(angle_diff) < 1e-6:
                    azi = self.start_azi
                else:
                    direction = 1.0 if angle_diff > 0 else -1.0
                    angular_delta_rad = dist_moved / max(dist, 0.1)
                    azi = self.start_azi + direction * np.degrees(angular_delta_rad)
                    azi = azi % 360
        else:
            # Traditional linear interpolation of progress
            if not self.is_circular:
                cur_x = sx + (ex - sx) * progress
                cur_y = sy + (ey - sy) * progress
                dist = np.sqrt(cur_x**2 + cur_y**2)
                azi = np.degrees(np.arctan2(cur_x, cur_y)) % 360
            else:
                # Linear interpolation in polar coordinates
                azi = self.start_azi + (self.end_azi - self.start_azi) * progress
                dist = self.start_dist + (self.end_dist - self.start_dist) * progress
                azi = azi % 360

        dist = np.clip(dist, DISTANCE_STEPS[0], DISTANCE_STEPS[-1])
        return azi, dist


if __name__ == "__main__":
    import soundfile as sf
    from config import DEFAULT_SAMPLE_RATE

    sr = DEFAULT_SAMPLE_RATE

    audio_path = r"mea.wav"
    output_path = "moving_sound.wav"

    print(f"Loading audio file: {audio_path}")
    if not os.path.exists(audio_path):
        print(f"Error: {audio_path} not found.")
    else:
        dry_data = load_and_resample(audio_path)

        sound = SpatialSound(dry_mono=dry_data, start_dist=3.0, start_azi=0.0, end_dist=3.0, end_azi=180.0)
        stereo_buffer = sound.compute_stereo()

        print(f"Writing to {output_path}...")
        sf.write(output_path, stereo_buffer.T, sr)
        print("Done!")
