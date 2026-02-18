import numpy as np
import dataset
from dataset import generate_epoch
import matplotlib.pyplot as plt

def test_moving_sounds():
    print("Testing moving sounds generation...")
    
    # Generate a small epoch
    # Force moving probability to 1.0 to ensure we get moving sounds
    chunks, labels = generate_epoch(
        total_duration_seconds=10,
        num_sounds=5,
        update_interval_ms=100,
        moving_prob=1.0,
        max_velocity=180.0,
        num_workers=2 # Use main process for debugging if possible, or 2
    )
    
    print(f"Generated chunks: {chunks.shape}")
    print(f"Generated labels: {labels.shape}")
    
    # Check if we have valid data
    assert chunks.shape[0] > 0
    assert not np.isnan(chunks).any()
    assert not np.isnan(labels).any()
    
    # Analyze labels for movement
    # For a moving sound, the peak of the label should shift over time.
    # We can detect this by finding the peak azimuth in each window.
    
    peak_azimuths = []
    for i in range(len(labels)):
        # Find index of max value
        idx = np.argmax(labels[i])
        # Convert to degrees (approx)
        azi = idx * (360 / labels.shape[1])
        peak_azimuths.append(azi)
        
    peak_azimuths = np.array(peak_azimuths)
    
    # Check if there is variation in peak azimuths
    # (Note: with only 5 sounds in 10 seconds, they might not overlap much, 
    # so we might see sequences of stable azimuths if sounds are short, 
    # or changing azimuths if they are long and moving).
    
    print(f"Peak azimuths (first 20): {peak_azimuths[:20]}")
    
    # Calculate difference between consecutive windows
    diffs = np.diff(peak_azimuths)
    
    # Filter out large jumps (new sound appearing)
    # Small non-zero diffs indicate movement.
    # wrap around 360 handling is needed but for simple check:
    small_moves = np.sum((np.abs(diffs) > 0) & (np.abs(diffs) < 20))
    
    print(f"Number of small azimuth shifts (indicating movement): {small_moves}")
    
    if small_moves > 0:
        print("SUCCESS: Detected movement in labels.")
    else:
        print("WARNING: No movement detected in labels. (Might be bad luck or bug)")

if __name__ == "__main__":
    test_moving_sounds()
