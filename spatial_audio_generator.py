import os
import numpy as np
import soundfile as sf

from HRTF_convolver import load_and_resample, SpatialSound
from config import DEFAULT_SAMPLE_RATE, SOUNDS_FOLDER, OUTPUT_FOLDER


if __name__ == "__main__":
    input_audio_path = os.path.join(SOUNDS_FOLDER, "mea.wav")
    output_audio_path = os.path.join(OUTPUT_FOLDER, "orbiting_sound.wav")
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

    # 3. Create SpatialSound object for orbiting (circular) motion
    # From 0 to 360 * num_rotations
    # We can calculate a velocity (v = omega * r) if we want it velocity-driven
    # v = (2 * pi * num_rotations / duration) * distance
    duration = len(dry_data) / DEFAULT_SAMPLE_RATE
    orbit_velocity = (2 * np.pi * num_rotations / duration) * orbit_distance

    sound = SpatialSound(
        dry_mono=dry_data,
        sr=DEFAULT_SAMPLE_RATE,
        start_dist=orbit_distance,
        start_azi=0.0,
        end_dist=orbit_distance,
        end_azi=360.0 * num_rotations,
        is_circular=True,
        speed=orbit_velocity,
    )

    # 4. Compute Stereo
    print(f"Generating orbiting sound... ({num_rotations} rotations at {orbit_distance}m)")
    stereo_buffer = sound.compute_stereo(normalize=True)

    # 5. Save
    print(f"Writing spatialized audio to {output_audio_path}")
    sf.write(output_audio_path, stereo_buffer.T, DEFAULT_SAMPLE_RATE)
    print("Done!")
