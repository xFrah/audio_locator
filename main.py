import slab
import numpy as np
import os

# Ensure output directory exists
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. Setup the environment and HRTF
# Load the built-in KEMAR dataset
hrtf = slab.HRTF.kemar()  
# Read your dry mono sound
dry_sound = slab.Sound.read(r'sounds\sounds\player\bodyshot_kill_01.wav')

# 2. Define coordinates for the dataset
angles = range(0, 360, 15)   # Azimuth 0-360 in 15 degree steps
distances = [1, 2, 3]        # Distances in meters

# 3. Create the virtual room
# Size: 10x10x3m. Listener centered at [5, 5, 1.5] for maximum clearance.
room_size = [10, 10, 3]
listener_pos = [5, 5, 1.5]

for dist in distances:
    for azi in angles:
        # Initialize room with listener at center
        room = slab.Room(size=room_size, listener=listener_pos)
        
        # 4. Set the source position using POLAR coordinates (azi, ele, dist)
        # This is relative to the listener, ensuring 'azi' matches exactly.
        # Elevation is set to 0 (ear level).
        room.set_source([azi, 0, dist])
        
        # 5. Compute the Room Impulse Response (RIR) including HRTF
        # The.hrir() method uses KEMAR by default to filter the echos
        hrir = room.hrir()
        
        # 6. Apply the spatial filter to the dry sound
        spatial_audio = hrir.apply(dry_sound)
        
        # 7. Save file with precise metadata in the filename
        filename = os.path.join(output_dir, f"gunshot_azi{azi}_dist{dist}.wav")
        spatial_audio.write(filename)

print(f"Dataset generation complete. Files saved to '{output_dir}'.")