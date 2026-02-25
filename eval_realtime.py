import os
import time
import socket
import librosa
import numpy as np
import torch
import matplotlib.cm as cm

from model import SpatialAudioHeatmapLocator
from config import *
from dataset import compute_spatial_features
from plot import LivePredictionPlot


class WLEDVisualizer:
    def __init__(self, host="wled.local", num_leds=120, width_ratio=16, height_ratio=9):
        self.num_leds = num_leds
        self.host = host
        self.port = 21324

        try:
            self.ip = socket.gethostbyname(host)
        except socket.gaierror:
            self.ip = host

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # DRGB protocol header: protocol 2, timeout 2 seconds
        self.header = bytearray([2, 2])

        # Precompute LED angles
        path_length = 2 * (width_ratio + height_ratio)
        half_w = width_ratio / 2
        half_h = height_ratio / 2

        angles = []
        for i in range(num_leds):
            pi = (i / num_leds) * path_length
            if pi < height_ratio:
                x, y = -half_w, -half_h + pi
            elif pi < height_ratio + width_ratio:
                x, y = -half_w + (pi - height_ratio), half_h
            elif pi < 2 * height_ratio + width_ratio:
                x, y = half_w, half_h - (pi - height_ratio - width_ratio)
            else:
                x, y = half_w - (pi - 2 * height_ratio - width_ratio), -half_h

            theta = np.arctan2(x, y)
            if theta < 0:
                theta += 2 * np.pi
            angles.append(theta)

        self.angles = np.array(angles)

    def update(self, pred_logits):
        pred_prob = 1.0 / (1.0 + np.exp(-pred_logits))
        azi_bins = pred_prob.shape[0]
        bin_angles = np.linspace(0, 2 * np.pi, azi_bins + 1)
        padded_prob = np.append(pred_prob, pred_prob[0])

        interpolated_prob = np.interp(self.angles, bin_angles, padded_prob)

        # Brightness: 0.4 -> 0.0, 1.0 -> 1.0
        brightness = np.clip((interpolated_prob - 0.4) / (1.0 - 0.4), 0.0, 1.0)

        red = (255 * brightness).astype(np.uint8)
        green = np.zeros_like(red, dtype=np.uint8)
        blue = np.zeros_like(red, dtype=np.uint8)

        # Layout GRB
        colors_grb = np.column_stack((green, red, blue))

        packet = bytearray(self.header)
        packet.extend(colors_grb.tobytes())

        try:
            self.sock.sendto(packet, (self.ip, self.port))
        except Exception:
            pass

    def stop(self):
        # Send one last black buffer to turn them off gracefully if real-time ends
        packet = bytearray(self.header)
        packet.extend(bytes([0, 0, 0] * self.num_leds))
        try:
            self.sock.sendto(packet, (self.ip, self.port))
            self.sock.close()
        except Exception:
            pass


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
    wled_plot = WLEDVisualizer(host="wled.local")

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
            wled_plot.update(pred_np)

            # Print occasional progress
            if (i + 1) % max(1, num_windows // 10) == 0 or i == 0:
                hz = 1.0 / chunk_duration
                print(f"Chunk {i+1}/{num_windows} - Process Time: {chunk_duration*1000:.1f}ms ({hz:.1f} Hz)")

    live_plot.stop()
    wled_plot.stop()

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
