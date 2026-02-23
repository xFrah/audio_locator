import os
import time
import librosa
import numpy as np
import torch

from model import SpatialAudioHeatmapLocator
from config import *
from dataset import compute_spatial_features
from plot import LivePredictionPlot


def evaluate_realtime(
    audio_path,
    model_path="model.pt",
    window_duration=DEFAULT_WINDOW_SIZE_SECONDS,
    hop_duration=0.1,  # e.g., 10 Hz refresh rate
    device=None,
    sr=DEFAULT_SAMPLE_RATE,
):
    if device is None:
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load Model ---
    model = SpatialAudioHeatmapLocator(
        input_channels=NUM_SPATIAL_CHANNELS,
        gcc_channels=NUM_GCC_CHANNELS,
        azi_bins=AZI_BINS,
    ).to(device)

    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            print("Successfully loaded model weights.")
        except Exception as e:
            print(f"Warning: Could not load model weights: {e}")
    else:
        print(f"Warning: {model_path} not found. Running with untrained weights.")

    model.eval()

    # --- Load Audio ---
    print(f"Loading audio from {audio_path}...")
    start_load = time.perf_counter()
    y, _ = librosa.load(audio_path, sr=sr, mono=False)
    load_time = time.perf_counter() - start_load
    print(f"Audio loaded in {load_time:.2f}s. Shape: {y.shape}")

    if y.ndim == 1:
        print("Warning: Audio is mono. Duplicating to stereo for processing.")
        y = np.stack((y, y), axis=0)
    elif y.shape[0] != 2:
        print(f"Error: Expected 2 channels, but audio has {y.shape[0]}. Exiting.")
        return

    # --- Real-time Simulation Parameters ---
    total_samples = y.shape[1]
    window_samples = int(window_duration * sr)
    hop_samples = int(hop_duration * sr)

    if total_samples < window_samples:
        print(f"Audio is too short ({total_samples} samples) for the window size ({window_samples} samples).")
        return

    num_windows = (total_samples - window_samples) // hop_samples + 1
    print(f"Total duration: {total_samples / sr:.2f}s")
    print(f"Config: Window={window_duration:.2f}s, Hop={hop_duration:.3f}s")
    print(f"Total chunks to process: {num_windows}")
    print("-" * 50)

    # --- Diagnostic variables ---
    times_processing = []

    print("Starting simulation (chunk by chunk)...")

    # --- Live Plot ---
    live_plot = LivePredictionPlot()

    # --- Warmup ---
    print("Warming up model and features...")
    with torch.no_grad():
        start_sample = 0
        end_sample = window_samples
        chunk = y[:, start_sample:end_sample]
        spatial_features, gcc_features = compute_spatial_features(chunk[0], chunk[1], sr=sr, n_fft=DEFAULT_N_FFT, hop_length=DEFAULT_HOP_LENGTH, n_gcc_bins=DEFAULT_GCC_MAX_TAU)
        x_spatial = torch.from_numpy(spatial_features).unsqueeze(0).to(device)
        x_gcc = torch.from_numpy(gcc_features).unsqueeze(0).to(device)
        for _ in range(3):
            out = model(x_spatial, x_gcc)
            _ = out.cpu().numpy()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    print("Warmup complete.")

    # --- Processing Loop (Simulating stream) ---
    with torch.no_grad():
        for i in range(num_windows):
            start_sample = i * hop_samples
            end_sample = start_sample + window_samples

            # The clock starts now for THIS chunk
            chunk_start_time = time.perf_counter()

            # 1. Grab the audio chunk (Simulates receiving audio from a buffer)
            chunk = y[:, start_sample:end_sample]

            # 2. Extract Features
            spatial_features, gcc_features = compute_spatial_features(chunk[0], chunk[1], sr=sr, n_fft=DEFAULT_N_FFT, hop_length=DEFAULT_HOP_LENGTH, n_gcc_bins=DEFAULT_GCC_MAX_TAU)

            # 3. To Tensor & Device
            x_spatial = torch.from_numpy(spatial_features).unsqueeze(0).to(device)
            x_gcc = torch.from_numpy(gcc_features).unsqueeze(0).to(device)

            # 4. Model Inference
            pred = model(x_spatial, x_gcc)

            # Simulate grabbing output back to CPU
            pred_np = pred.squeeze(0).cpu().numpy()

            # Clock stops for this chunk
            chunk_end_time = time.perf_counter()

            chunk_duration = chunk_end_time - chunk_start_time
            times_processing.append(chunk_duration)

            # Update live plot
            current_time_sec = (i + 1) * hop_duration
            live_plot.update(pred_np, current_time_sec)

            # Print occasional progress
            if (i + 1) % max(1, num_windows // 10) == 0 or i == 0:
                hz = 1.0 / chunk_duration
                print(f"Chunk {i+1}/{num_windows} - Process Time: {chunk_duration*1000:.1f}ms ({hz:.1f} Hz)")

    live_plot.stop()

    # --- Summary ---
    print("-" * 50)
    print("Real-time Processing Summary:")
    avg_time = np.mean(times_processing)
    min_time = np.min(times_processing)
    max_time = np.max(times_processing)
    p95_time = np.percentile(times_processing, 95)

    print(f"Chunks processed      : {len(times_processing)}")
    print(f"Average time per chunk: {avg_time*1000:.2f} ms")
    print(f"Min time per chunk    : {min_time*1000:.2f} ms")
    print(f"p95 time per chunk    : {p95_time*1000:.2f} ms")
    print(f"Max time per chunk    : {max_time*1000:.2f} ms")
    print(f"Average Frequency     : {1.0 / avg_time:.1f} Hz")

    if 1.0 / avg_time > (1.0 / hop_duration):
        print(f"Result: SUCCESS. Model runs faster than the required hop rate of {1.0 / hop_duration:.1f} Hz.")
    else:
        print(f"Result: WARNING. Model ({1.0/avg_time:.1f} Hz) is slower than the required hop rate ({1.0/hop_duration:.1f} Hz).")


if __name__ == "__main__":
    evaluate_realtime(
        audio_path=os.path.join(OUTPUT_FOLDER, "orbiting_sound.wav"),
        model_path=os.path.join(MODELS_FOLDER, "best.pt"),
        window_duration=2.0,
        hop_duration=0.1,
        device="cuda:0",
    )
