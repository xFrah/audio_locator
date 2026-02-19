
import sys
import numpy as np
import dataset

def verify_dataset():
    print("Verifying dataset labeling changes...")
    
    # Generate a very small epoch
    # 10 seconds, 10 sounds. High probability of overlap.
    chunks, labels, meta = dataset.generate_epoch(
        total_duration_seconds=10,
        num_sounds=20,
        window_size_seconds=0.5,
        update_interval_ms=100, # 100ms hop
        num_workers=4
    )
    
    print(f"Chunks shape: {chunks.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Metadata length: {len(meta)}")
    
    if len(meta) > 0:
        print("Sample metadata for window 0:")
        print(meta[0])
    
    # Check 1: Label range
    l_min = labels.min()
    l_max = labels.max()
    print(f"Label range: [{l_min:.4f}, {l_max:.4f}]")
    
    if l_max > 1.0:
        print("ERROR: Label > 1.0 found!")
    if l_min < 0.0:
        print("ERROR: Label < 0.0 found!")
        
    # Check 2: Check for presence of "silent" windows (should be NONE)
    sums = labels.sum(axis=1)
    active_count = np.sum(sums > 0)
    print(f"Active windows: {active_count} / {len(labels)}")
    
    if active_count != len(labels):
        print("ERROR: Silent windows found! Filtering failed.")
    else:
        print("SUCCESS: All windows contain active sound.")
    
    # Check 3: Check distinct intensity levels
    # We expect values >= 0.5 for active peaks.
    peaks = labels.max(axis=1)
    active_peaks = peaks[peaks > 0]
    if len(active_peaks) > 0:
        print(f"Active peak range: [{active_peaks.min():.4f}, {active_peaks.max():.4f}]")
        if active_peaks.min() < 0.5:
             print("WARNING: Active peak < 0.5 found! (Should be impossible with new logic 0.5 + ...)")
    
    print("Verification complete.")

if __name__ == "__main__":
    verify_dataset()
