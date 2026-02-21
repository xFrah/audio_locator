import os
import random
import numpy as np
import scipy.io
import soundfile as sf
import HRTF_convolver
import wave

from convert_wav import DEFAULT_SAMPLE_RATE

def generate_moving_sound(duration_sec=40.0, output_path="moving_sound.wav", audio_path=r"mea.wav"):
    sr = DEFAULT_SAMPLE_RATE
    
    print(f"Loading audio file: {audio_path}")
    if not os.path.exists(audio_path):
        print(f"Error: {audio_path} not found.")
        return
        
    dry_data = HRTF_convolver.load_and_resample(audio_path, target_sr=sr)
    
    # Loop the sound to reach the target duration
    target_samples = int(duration_sec * sr)
    if len(dry_data) < target_samples:
        repeats = int(np.ceil(target_samples / len(dry_data)))
        dry_data = np.tile(dry_data, repeats)
    
    # Fade out at the very end to avoid clicking
    dry_data = dry_data[:target_samples]
    fade_len = int(0.1 * sr)
    if len(dry_data) > fade_len:
        dry_data[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)
    
    n_samples = len(dry_data)
    
    print("Loading HRIR data from CIPIC subject 003...")
    subject = '003'
    hrir_mat = scipy.io.loadmat('CIPIC_hrtf_database/standard_hrir_database/subject_' + subject + '/hrir_final.mat')
    hrir_l = np.array(hrir_mat['hrir_l'])
    hrir_r = np.array(hrir_mat['hrir_r'])

    # Overlap-save algorithm parameters from HRTF_convolver.py
    L = 512 # Window size for the input chunk
    M = 200  # HRIR length
    N = int(2**np.ceil(np.log2(np.abs(L+M-1)))) # DFT size
    
    # Adjusted L so that N is a power of 2
    L = N - M + 1 
    
    buffer_OLAP_L = np.zeros(M - 1)
    buffer_OLAP_R = np.zeros(M - 1)
    
    # We will accumulate output here. Since output is same length as input + reverb tail (which we can truncate)
    out_length = n_samples
    spatial_L = np.zeros(out_length, dtype=np.float32)
    spatial_R = np.zeros(out_length, dtype=np.float32)

    print(f"Generating movement over {duration_sec} seconds using overlap-save...")
    
    start_azi = random.uniform(min(HRTF_convolver.AZIMUTH_ANGLES), max(HRTF_convolver.AZIMUTH_ANGLES))
    # Keep elevation near 0
    curr_el = 0.0 
    
    # Filter the entire audio track offline to prevent zero-phase chunking edge artifacts
    cutoff_freq = 200.0
    dry_data_lp = HRTF_convolver.butter_lp_filter(signal=dry_data, cutoff=cutoff_freq)
    dry_data = HRTF_convolver.butter_hp_filter(signal=dry_data, cutoff=cutoff_freq)
    
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
        curr_azi = np.sin(2 * np.pi * t_center / 12.0) * max(HRTF_convolver.AZIMUTH_ANGLES)
        
        # Interpolate to get the HRIR at the position selected
        h_l, h_r = HRTF_convolver.interp_hrir(
            triang=HRTF_convolver.TRI,
            points=HRTF_convolver.POINTS,
            T_inv=HRTF_convolver.T_INV,
            hrir_l=hrir_l, hrir_r=hrir_r,
            azimuth=curr_azi, elevation=curr_el
        )
        
        # Compute DFT of impulse response
        h = np.vstack(([h_l, h_r]))
        h = np.hstack((h, np.zeros((2, N-(M-1)))))
        H = np.fft.fft(h, N)
        
        # Overlap Save Algorithm on high frequencies
        x_L_overlap = np.hstack((buffer_OLAP_L, x_r))
        x_L_zeropad = np.hstack((x_L_overlap, np.zeros(N - len(x_L_overlap))))
        
        x_R_overlap = np.hstack((buffer_OLAP_R, x_r))
        x_R_zeropad = np.hstack((x_R_overlap, np.zeros(N - len(x_R_overlap))))
        
        # Save overlap for next iteration
        buffer_OLAP_L[:] = x_L_zeropad[N-(M-1):N]
        buffer_OLAP_R[:] = x_R_zeropad[N-(M-1):N]
        
        # Convolution using high frequencies only
        Xm_L = np.fft.fft(x_L_zeropad, N)
        Xm_R = np.fft.fft(x_R_zeropad, N)
        
        Ym_L = Xm_L * H[0]
        Ym_R = Xm_R * H[1]
        
        ym_L = np.real(np.fft.ifft(Ym_L))
        ym_R = np.real(np.fft.ifft(Ym_R))
        
        # Add back omnidirectional low frequencies (direct time-aligned addition)
        l_out = ym_L[M-1:N+1] + x_r_lp # First M-1 samples are Aliased/Discarded
        r_out = ym_R[M-1:N+1] + x_r_lp
        
        # Determine how much to write
        write_len = min(L, n_samples - b_start)
        
        if write_len > 0:
            spatial_L[b_start:b_start+write_len] = l_out[:write_len]
            spatial_R[b_start:b_start+write_len] = r_out[:write_len]
            
    # Normalize
    stereo_buffer = np.stack((spatial_L, spatial_R), axis=0) # (2, len)
    peak = np.max(np.abs(stereo_buffer))
    if peak > 0:
        stereo_buffer = stereo_buffer / peak * 0.9
        
    print(f"Writing to {output_path}...")
    sf.write(output_path, stereo_buffer.T, sr)
    print("Done!")

if __name__ == "__main__":
    generate_moving_sound()
