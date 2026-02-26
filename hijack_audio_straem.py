import soundcard as sc
import numpy as np
import warnings

# Suppress the "data discontinuity in recording" warning from soundcard
warnings.filterwarnings("ignore", category=sc.SoundcardRuntimeWarning)


def listen_to_windows_audio():
    # 1. Get the default Windows speaker
    speaker = sc.default_speaker()
    print(f"Default speaker found: {speaker.name}")

    # 2. Get the loopback interface for that specific speaker
    # include_loopback=True is the magic parameter that captures output
    loopback = sc.get_microphone(id=speaker.name, include_loopback=True)
    print(f"Listening to loopback: {loopback.name}...\n")

    # 3. Open the stream and process it in real-time
    # samplerate is usually 44100 Hz or 48000 Hz
    with loopback.recorder(samplerate=48000) as mic:
        try:
            while True:
                # Capture a small chunk of audio data
                # numframes determines the buffer size (lower = lower latency)
                data = mic.record(numframes=1024)

                # 'data' is a 2D numpy array [frames, channels]
                # --- INSERT YOUR REAL-TIME PROCESSING HERE ---

                # Example: Calculate the Root Mean Square (RMS) to measure volume
                rms = np.sqrt(np.mean(data**2))

                # Create a simple visual volume meter in the console
                meter = "|" * int(rms * 200)
                print(f"Vol: {rms:.4f} {meter}".ljust(50), end="\r")

        except KeyboardInterrupt:
            print("\nStopped listening.")


if __name__ == "__main__":
    listen_to_windows_audio()
