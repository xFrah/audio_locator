import torch
import warnings
import os
import threading
import time
import gc

# Suppress "PyTorch is not compiled with NCCL support" warning on Windows
warnings.filterwarnings("ignore", message=".*PyTorch is not compiled with NCCL support.*")
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

from main import SpatialAudioHeatmapLocator
from convert_wav import NUM_FEATURE_CHANNELS
from dataset import generate_epoch


def bce_loss(pred_logits, target):
    """BCE with dynamic pos_weight = N/K so predicting all-zero is not free."""
    num_elements = target.numel()
    num_pos = target.sum().clamp(min=1)
    pos_weight = (num_elements / num_pos).clamp(max=100.0)  # cap to avoid explosion
    return F.binary_cross_entropy_with_logits(
        pred_logits, target,
        pos_weight=pos_weight.expand_as(target),
    )


class LiveComparisonPlot:
    """Persistent matplotlib window that updates GT vs Predicted each epoch."""

    def __init__(self):
        import multiprocessing as mp
        self._q = mp.Queue(maxsize=1)
        self._p = mp.Process(target=self._run_viewer, args=(self._q,), daemon=True)
        self._p.start()

    def stop(self):
        try:
            self._q.put(None, timeout=1)
        except:
            pass
        self._p.join(timeout=2.0)
            
    def pump_events(self):
        pass # Deprecated, handled by background process

    def update(self, gt, pred_logits, metadata, epoch):
        import queue
        if self._q.full():
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
        try:
            self._q.put((gt, pred_logits, metadata, epoch), timeout=0.1)
        except queue.Full:
            pass

    @staticmethod
    def _run_viewer(q):
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        import numpy as np
        import queue
        
        plt.ion()
        fig = plt.figure(figsize=(14, 10))  # Taller figure for 2x2
        fig.suptitle("Waiting for first epoch…", fontsize=15)
        fig.canvas.draw()
        plt.pause(0.01)
        
        while True:
            try:
                data = q.get(timeout=0.05)
                if data is None:
                    break
                gt, pred_logits, metadata, epoch = data

                pred_prob = 1.0 / (1.0 + np.exp(-pred_logits))

                azi_bins = gt.shape[0]
                azimuths = np.linspace(0, 2 * np.pi, azi_bins, endpoint=False)
                width = 2 * np.pi / azi_bins

                fig.clf()

                # --- 1. Ground Truth (Polar) ---
                ax_gt = fig.add_subplot(2, 2, 1, projection="polar")
                colors = plt.cm.magma(gt)
                ax_gt.bar(azimuths, np.ones_like(gt), width=width, bottom=0.0,
                       color=colors, alpha=0.9)
                ax_gt.set_ylim(0, 1)
                ax_gt.set_theta_zero_location("N")
                ax_gt.set_theta_direction(-1)
                ax_gt.set_title("Ground Truth", fontsize=13, pad=12)

                # --- 2. Predicted (Polar) ---
                ax_pred = fig.add_subplot(2, 2, 2, projection="polar")
                colors = plt.cm.magma(pred_prob)
                ax_pred.bar(azimuths, np.ones_like(pred_prob), width=width, bottom=0.0,
                       color=colors, alpha=0.9)
                ax_pred.set_ylim(0, 1)
                ax_pred.set_theta_zero_location("N")
                ax_pred.set_theta_direction(-1)
                ax_pred.set_title("Predicted", fontsize=13, pad=12)

                # --- 3. Room Map (Cartesian) ---
                ax_map = fig.add_subplot(2, 2, 3)
                ax_map.set_title("Room Map (Top-Down)")
                ax_map.set_xlim(-5, 5) # Assuming room size ~10m
                ax_map.set_ylim(-5, 5)
                ax_map.set_aspect('equal')
                ax_map.grid(True, alpha=0.3)
                
                # Plot listener
                ax_map.plot(0, 0, 'k+', markersize=10, label="Listener")
                
                # Plot sounds
                for m in metadata:
                    curr_azi_rad = np.radians(m['current_pos'][0])
                    curr_x = m['current_pos'][1] * np.sin(curr_azi_rad)
                    curr_y = m['current_pos'][1] * np.cos(curr_azi_rad)
                    
                    # Check if moving (tolerance for float equality)
                    is_moving = (abs(m['traj_start'][0] - m['traj_end'][0]) > 0.1 or 
                                 abs(m['traj_start'][1] - m['traj_end'][1]) > 0.1)
                    
                    if is_moving:
                        color = 'r' # Moving = Red
                        
                        # Draw straight trajectory (Cartesian)
                        # Since movement is now linear in Cartesian space, we just draw a line.
                        
                        # Convert start/end to Cartesian for plotting
                        start_azi_rad = np.radians(m['traj_start'][0])
                        start_x = m['traj_start'][1] * np.sin(start_azi_rad)
                        start_y = m['traj_start'][1] * np.cos(start_azi_rad)
                        
                        # Draw from Start -> Current (Past only)
                        ax_map.plot([start_x, curr_x], [start_y, curr_y], '-', color=color, alpha=0.3)
                        
                    else:
                        color = 'b' # Static = Blue
                    
                    # Draw current pos
                    ax_map.plot(curr_x, curr_y, 'o', color=color)
                    ax_map.text(curr_x + 0.2, curr_y, str(m['id']), fontsize=8, color=color)
                    
                    # Retrieve physical sound properties
                    radius = m.get('radius', 1.0)
                    width_deg = m.get('width_deg', 0.0)

                    # Draw the physical radius around the sound position
                    circle = matplotlib.patches.Circle((curr_x, curr_y), radius, fill=False, color=color, alpha=0.5, linestyle='--')
                    ax_map.add_patch(circle)

                    # Draw the FOV projection cone from listener (0,0) to sound
                    if m['current_pos'][1] > radius:
                        # FOV bounded tightly by radius tangents
                        half_angle_rad = np.radians(width_deg / 2)
                        # left tangent
                        angle_left = curr_azi_rad - half_angle_rad
                        # right tangent
                        angle_right = curr_azi_rad + half_angle_rad
                        
                        tangent_dist = np.sqrt(max(0, m['current_pos'][1]**2 - radius**2))
                        lx_tangent = tangent_dist * np.sin(angle_left)
                        ly_tangent = tangent_dist * np.cos(angle_left)
                        rx_tangent = tangent_dist * np.sin(angle_right)
                        ry_tangent = tangent_dist * np.cos(angle_right)
                        
                        ax_map.plot([0, lx_tangent], [0, ly_tangent], color='green', alpha=0.3)
                        ax_map.plot([0, rx_tangent], [0, ry_tangent], color='green', alpha=0.3)
                        
                    else:
                        # Listener is inside the circle
                        ax_map.plot(0, 0, marker='o', markersize=30, markeredgecolor='green', markerfacecolor='none', alpha=0.3)

                # --- 4. Timeline (Gantt) ---
                ax_time = fig.add_subplot(2, 2, 4)
                ax_time.set_title("Timeline (Relative Time [s])")
                
                if metadata:
                    win_start, win_end = metadata[0]['win_range']
                    sr = 44100 # Assuming default SR
                    
                    # Center view around window
                    # Window duration in samples
                    win_duration_samples = win_end - win_start
                    
                    yticks = []
                    yticklabels = []
                    
                    for i, m in enumerate(metadata):
                        y = i
                        # Convert active samples to relative time (seconds)
                        # t=0 is the start of the current window
                        start_rel_sec = (m['start_sample'] - win_start) / sr
                        end_rel_sec = (m['end_sample'] - win_start) / sr
                        duration_sec = end_rel_sec - start_rel_sec
                        
                        ax_time.barh(y, duration_sec, left=start_rel_sec, height=0.6, align='center', color='green', alpha=0.6)
                        yticks.append(y)
                        yticklabels.append(f"Sound {m['id']}")
                    
                    ax_time.set_yticks(yticks)
                    ax_time.set_yticklabels(yticklabels)
                    
                    # Highlight current window [0, win_duration_in_sec]
                    win_duration_sec = win_duration_samples / sr
                    ax_time.axvspan(0, win_duration_sec, color='red', alpha=0.1, label='Current Window')
                    ax_time.axvline(win_duration_sec, color='red', linestyle='--', label='Instant')
                    
                    # Set x-limit to show window + context
                    # Show a bit of context before and after
                    context = 1.0 # 1 second context
                    ax_time.set_xlim(-context, win_duration_sec + context)
                    ax_time.set_xlabel("Time (s) [0 = Window Start]")

                fig.suptitle(f"Epoch {epoch} — Sample", fontsize=15)
                fig.tight_layout()
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            except queue.Empty:
                pass
            except Exception as e:
                pass
            
            plt.pause(0.01)


def train(epochs=100,
          batch_size=12,
          lr=1e-5,
          azi_bins=180,
          epoch_duration_seconds=5000,
          device=None):

    if device is None:
        # device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Model, optimizer ---
    model = SpatialAudioHeatmapLocator(
        input_channels=NUM_FEATURE_CHANNELS, azi_bins=azi_bins
    ).to(device)

    if os.path.exists("resume.pt"):
        print("Found resume.pt, resuming training from these weights...")
        state_dict = torch.load("resume.pt", map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    # Use DataParallel if multiple GPUs are available
    # if torch.cuda.device_count() > 1:
    #     print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
    #     model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- Live plot ---
    live_plot = LiveComparisonPlot()

    # --- Data generation config ---
    num_sounds = int(epoch_duration_seconds * 1)
    gen_kwargs = dict(total_duration_seconds=epoch_duration_seconds,
                      num_sounds=num_sounds, update_interval_ms=2000, max_velocity=90, moving_prob=0.8)

    # --- Training loop ---
    best_loss = float('inf')
    
    for epoch in tqdm(range(epochs), desc="Epochs"):

        # Generate this epoch's data
        tqdm.write(f"\nGenerating epoch {epoch+1} data ({epoch_duration_seconds}s, {num_sounds} sounds)…")
        train_chunks, train_labels, train_metadata = generate_epoch(**gen_kwargs)
        train_chunks = torch.from_numpy(train_chunks)
        train_labels = torch.from_numpy(train_labels)
        tqdm.write(f"--- Epoch {epoch+1}: {len(train_chunks)} train samples ready")

        # Shuffle
        perm = torch.randperm(len(train_chunks))
        train_chunks = train_chunks[perm]
        train_labels = train_labels[perm]
        train_metadata = [train_metadata[i] for i in perm.tolist()]

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_batches = 0

        batch_bar = tqdm(range(0, len(train_chunks), batch_size),
                         desc=f"Epoch {epoch+1} train", leave=False)
        for start in batch_bar:
            end = min(start + batch_size, len(train_chunks))
            x = train_chunks[start:end].to(device)
            y = train_labels[start:end].to(device)

            pred = model(x)
            loss = bce_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1
            batch_bar.set_postfix(loss=f"{loss.item():.6f}")

            # Update live plot with last sample of this batch
            # Update every batch as requested by user
            live_plot.update(
                y[-1].cpu().numpy(),
                pred[-1].detach().cpu().numpy(),
                train_metadata[end-1], # Sample metadata
                epoch + 1,
            )

        avg_train = train_loss / train_batches
        tqdm.write(f"Epoch {epoch+1:3d}/{epochs}  train_loss={avg_train:.6f}")
        
        # Save if best
        if avg_train < best_loss:
            best_loss = avg_train
            save_path = "model.pt"
            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(state_dict, save_path)
            tqdm.write(f"New best loss! Saved model to {save_path}")

        # --- Explicit Garbage Collection to prevent RAM spikes ---
        del train_chunks
        del train_labels
        del train_metadata
        gc.collect()

    live_plot.stop()
    print(f"\nTraining complete. Best loss: {best_loss:.6f}")


if __name__ == "__main__":
    train()

