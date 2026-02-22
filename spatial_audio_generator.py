import os
import librosa
import numpy as np
import soundfile as sf
import scipy.signal

import HRTF_convolver
from HRTF_convolver import load_and_resample, load_hrir_cache, TRI, POINTS, T_INV, DISTANCE_STEPS, butter_hp_filter, butter_lp_filter, interp_hrir
from convert_wav import DEFAULT_SAMPLE_RATE


def generate_orbiting_sound(dry_data, sr, distance=3.0, rotations=1.0, normalize=True):  # Number of full 360 orbits to complete over the file length
    """
    Applies a moving HRTF filter to a mono audio signal, simulating the sound
    orbiting around the listener's head.

    This uses the block-based overlap-add convolution logic from HRTF_convolver.py,
    interpolating along polar coordinates.
    """
    n_samples = len(dry_data)

    if HRTF_convolver._hrir_shm_meta is None:
        raise RuntimeError("HRIR cache not initialized. Call load_hrir_cache() first.")

    hrir_cache = HRTF_convolver._hrir_shm_meta
    first_val = next(iter(hrir_cache.values()))

    # Check if from disk or shared memory
    if len(first_val) == 3:
        M = first_val[2]
    else:
        M = len(first_val[0])

    out_length = n_samples + M - 1
    spatial_L = np.zeros(out_length, dtype=np.float32)
    spatial_R = np.zeros(out_length, dtype=np.float32)

    cutoff_freq = 200.0
    dry_data_lp = butter_lp_filter(signal=dry_data, cutoff=cutoff_freq)
    dry_data_hp = butter_hp_filter(signal=dry_data, cutoff=cutoff_freq)

    window_size = 2048
    hop = window_size // 2
    window = scipy.signal.get_window("hann", window_size).astype(np.float32)

    # Orbiting logic: Azimuth goes from 0 to 360 * rotations
    start_azi = 0.0
    end_azi = 360.0 * rotations

    print(f"Generating orbiting sound... ({rotations} rotations at {distance}m)")

    for b_start in range(0, n_samples - window_size + 1, hop):
        b_end = b_start + window_size
        x_r = dry_data_hp[b_start:b_end] * window
        x_r_lp = dry_data_lp[b_start:b_end] * window

        # Calculate progress [0.0, 1.0]
        progress = (b_start + hop) / n_samples
        progress = np.clip(progress, 0.0, 1.0)

        # Linearly interpolate azimuth
        current_azi = start_azi + (end_azi - start_azi) * progress
        # Wrap azimuth to [0, 360) for the lookup
        current_azi_wrapped = current_azi % 360.0

        # HRTF lookup
        h_l_next, h_r_next = interp_hrir(triang=TRI, points=POINTS, T_inv=T_INV, hrir_dict=hrir_cache, azimuth=current_azi_wrapped, distance=distance)

        conv_L_cross = scipy.signal.fftconvolve(x_r, h_l_next, mode="full")
        conv_R_cross = scipy.signal.fftconvolve(x_r, h_r_next, mode="full")

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


if __name__ == "__main__":
    input_audio_path = r"mea.wav"
    output_audio_path = r"orbiting_sound.wav"
    orbit_distance = 3.0
    num_rotations = 0.5

    # Ensure output directory exists
    output_dir = os.path.dirname(output_audio_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading input audio from {input_audio_path}")
    if not os.path.exists(input_audio_path):
        print(f"Error: Could not find '{input_audio_path}'")
        exit(1)

    # 1. Load Audio
    dry_data = load_and_resample(input_audio_path, target_sr=DEFAULT_SAMPLE_RATE)

    # 2. Load HRIR Cache
    load_hrir_cache()

    # 3. Apply Orbiting HRTF
    stereo_buffer = generate_orbiting_sound(dry_data=dry_data, sr=DEFAULT_SAMPLE_RATE, distance=orbit_distance, rotations=num_rotations)

    # 4. Save
    print(f"Writing spatialized audio to {output_audio_path}")
    sf.write(output_audio_path, stereo_buffer.T, DEFAULT_SAMPLE_RATE)
    print("Done!")
