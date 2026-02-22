import os
import librosa
import numpy as np
import soundfile as sf
import scipy.signal

import HRTF_convolver
from HRTF_convolver import load_and_resample, load_hrir_cache, SpatialSound
from convert_wav import DEFAULT_SAMPLE_RATE


if __name__ == "__main__":
    input_audio_path = r"mea.wav"
    output_audio_path = r"orbiting_sound.wav"
    orbit_distance = 3.0
    num_rotations = 3.0

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

    # 3. Create SpatialSound object for orbiting (circular) motion
    # From 0 to 360 * num_rotations
    sound = SpatialSound(
        dry_mono=dry_data, sr=DEFAULT_SAMPLE_RATE, start_dist=orbit_distance, start_azi=0.0, end_dist=orbit_distance, end_azi=360.0 * num_rotations, is_circular=True
    )

    # 4. Compute Stereo
    print(f"Generating orbiting sound... ({num_rotations} rotations at {orbit_distance}m)")
    stereo_buffer = sound.compute_stereo(normalize=True)

    # 5. Save
    print(f"Writing spatialized audio to {output_audio_path}")
    sf.write(output_audio_path, stereo_buffer.T, DEFAULT_SAMPLE_RATE)
    print("Done!")
