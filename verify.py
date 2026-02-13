import binaspect
import librosa
import matplotlib.pyplot as plt

# Load one of your generated files (ensure it's loaded as stereo)
audio_path = r'output\gunshot_azi45_dist2.wav'
audio, sr = librosa.load(audio_path, sr=44100, mono=False)

# 1. Verify ITD (Timing cues)
# Looking for a sharp ridge at the delay corresponding to 45 degrees
binaspect.ITD_hist(audio, sr, plots=True)
plt.title("ITD (Timing) Verification")

# 2. Verify ILR (Level/Intensity cues)
# Looking for a ridge showing higher volume in the leading ear
binaspect.ILR_hist(audio, sr, plots=True)
plt.title("ILR (Level) Verification")

plt.show()