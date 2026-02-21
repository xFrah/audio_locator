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
    L = 2048 # Window size for the input chunk
    M = 200  # HRIR length
    N = int(2**np.ceil(np.log2(np.abs(L+M-1)))) # DFT size
    
    # Adjusted L so that N is a power of 2
    L = N - M + 1 
    
    buffer_OLAP = np.zeros(M - 1)
    
    # We will accumulate output here. Since output is same length as input + reverb tail (which we can truncate)
    out_length = n_samples
    spatial_L = np.zeros(out_length, dtype=np.float32)
    spatial_R = np.zeros(out_length, dtype=np.float32)

    print(f"Generating movement over {duration_sec} seconds using overlap-save...")
    
    start_azi = random.uniform(min(HRTF_convolver.AZIMUTH_ANGLES), max(HRTF_convolver.AZIMUTH_ANGLES))
    # Keep elevation near 0
    curr_el = 0.0 
    
    for b_start in range(0, n_samples, L):
        b_end = min(b_start + L, n_samples)
        x_r = dry_data[b_start:b_end]
        
        # If last block is shorter than L, pad it with zeros
        padded_chunk = False
        if len(x_r) < L:
            x_r = np.pad(x_r, (0, L - len(x_r)))
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
        
        # Overlap Save Algorithm
        x_r_overlap = np.hstack((buffer_OLAP, x_r))
        x_r_zeropad = np.hstack((x_r_overlap, np.zeros(N - len(x_r_overlap))))
        
        # Save overlap for next iteration
        buffer_OLAP[:] = x_r_zeropad[N-(M-1):N]
        
        # Crossover (low frequencies are non-directional)
        x_r_zeropad_lp = HRTF_convolver.butter_lp_filter(signal=x_r_zeropad, cutoff=200, fs=sr)
        x_r_zeropad_hp = HRTF_convolver.butter_hp_filter(signal=x_r_zeropad, cutoff=200, fs=sr)
        
        # Convolution
        Xm = np.tile(np.fft.fft(x_r_zeropad_hp, N), (2, 1))
        Ym = Xm * H
        ym = np.real(np.fft.ifft(Ym))
        
        l_out = ym[0, M-2:N] + x_r_zeropad_lp[M-2:N]
        r_out = ym[1, M-2:N] + x_r_zeropad_lp[M-2:N]
        
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
