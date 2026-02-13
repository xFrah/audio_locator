import slab
import numpy as np

# 1. Setup the environment and HRTF
hrtf = slab.HRTF.kemar()  # Use built-in mannequin data
dry_sound = slab.Sound.read(r'sounds\sounds\player\bodyshot_kill_01.wav')

# 2. Define coordinates to sample (Azimuth 0-360 in 15 degree steps)
angles = range(0, 360, 15)
distances = [1, 2, 3]  # Distances in meters (reduced to fit within room bounds)

for dist in distances:
    for azi in angles:
        # 3. Create a virtual room (size in meters)
        # Places listener at [2, 2, 1.5] in a 5x5x3m room
        room = slab.Room(size=[5, 5, 3], listener=[2, 2, 1.5])
        
        # 4. Set the source position (Cartesian: x, y, z)
        # Convert from polar (azi, elevation=0, dist) to Cartesian coordinates
        # Place source relative to room center [2.5, 2.5, 1.5]
        rad = np.deg2rad(azi)
        x = 2.5 + dist * np.cos(rad)
        y = 2.5 + dist * np.sin(rad)
        z = 1.5
        room.set_source([x, y, z])
        
        # 5. Compute the Room Impulse Response (RIR) including HRTF
        hrir = room.hrir()
        
        # 6. Apply to the dry sound
        spatial_audio = hrir.apply(dry_sound)
        
        # 7. Write to file with metadata in the filename
        filename = rf"output\gunshot_azi{azi}_dist{dist}.wav"
        spatial_audio.write(filename)
