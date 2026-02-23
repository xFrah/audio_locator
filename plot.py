import multiprocessing as mp
import queue
import numpy as np
import HRTF_convolver


class LiveComparisonPlot:
    """Persistent matplotlib window that updates GT vs Predicted each epoch."""

    def __init__(self):
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
        pass  # Deprecated, handled by background process

    def update(self, gt, pred_logits, metadata, epoch):
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
                ax_gt.bar(azimuths, np.ones_like(gt), width=width, bottom=0.0, color=colors, alpha=0.9)
                ax_gt.set_ylim(0, 1)
                ax_gt.set_theta_zero_location("N")
                ax_gt.set_theta_direction(-1)
                ax_gt.set_title("Ground Truth", fontsize=13, pad=12)

                # --- 2. Predicted (Polar) ---
                ax_pred = fig.add_subplot(2, 2, 2, projection="polar")
                colors = plt.cm.magma(pred_prob)
                ax_pred.bar(azimuths, np.ones_like(pred_prob), width=width, bottom=0.0, color=colors, alpha=0.9)
                ax_pred.set_ylim(0, 1)
                ax_pred.set_theta_zero_location("N")
                ax_pred.set_theta_direction(-1)
                ax_pred.set_title("Predicted", fontsize=13, pad=12)

                # --- 3. Room Map (Cartesian) ---
                ax_map = fig.add_subplot(2, 2, 3)
                ax_map.set_title("Room Map (Top-Down)")
                ax_map.set_xlim(-5, 5)  # Assuming room size ~10m
                ax_map.set_ylim(-5, 5)
                ax_map.set_aspect("equal")
                ax_map.grid(True, alpha=0.3)

                # Plot listener
                ax_map.plot(0, 0, "k+", markersize=10, label="Listener")

                # Plot sounds
                for m in metadata:
                    curr_azi_rad = np.radians(m["current_pos"][0])
                    curr_x = m["current_pos"][1] * np.sin(curr_azi_rad)
                    curr_y = m["current_pos"][1] * np.cos(curr_azi_rad)

                    # Retrieve sound properties from the object
                    sound_obj = m["sound_obj"]
                    is_moving = not sound_obj.is_stationary

                    if is_moving:
                        color = "r"  # Moving = Red

                        # Calculate current progress based on window end relative to sound duration
                        progress = (m["win_range"][1] - m["start_sample"]) / (m["end_sample"] - m["start_sample"])
                        progress = np.clip(progress, 0.0, 1.0)

                        # Compute trajectory points on-the-fly for visualization
                        n_traj_steps = 50
                        traj_pts = [sound_obj.get_pos(p) for p in np.linspace(0, progress, n_traj_steps)]

                        if traj_pts:
                            path_x = [p_dist * np.sin(np.radians(p_azi)) for p_azi, p_dist in traj_pts]
                            path_y = [p_dist * np.cos(np.radians(p_azi)) for p_azi, p_dist in traj_pts]
                            ax_map.plot(path_x, path_y, "-", color=color, alpha=0.3)
                    else:
                        color = "b"  # Static = Blue

                    # Draw current pos
                    ax_map.plot(curr_x, curr_y, "o", color=color)
                    ax_map.text(curr_x + 0.2, curr_y, str(m["id"]), fontsize=8, color=color)

                    # Retrieve physical sound properties
                    radius = m.get("radius", 1.0)
                    width_deg = m.get("width_deg", 0.0)

                    # Draw the physical radius around the sound position
                    circle = matplotlib.patches.Circle((curr_x, curr_y), radius, fill=False, color=color, alpha=0.5, linestyle="--")
                    ax_map.add_patch(circle)

                    # Draw the FOV projection cone from listener (0,0) to sound
                    if m["current_pos"][1] > radius:
                        # FOV bounded tightly by radius tangents
                        half_angle_rad = np.radians(width_deg / 2)
                        # left tangent
                        angle_left = curr_azi_rad - half_angle_rad
                        # right tangent
                        angle_right = curr_azi_rad + half_angle_rad

                        tangent_dist = np.sqrt(max(0, m["current_pos"][1] ** 2 - radius**2))
                        lx_tangent = tangent_dist * np.sin(angle_left)
                        ly_tangent = tangent_dist * np.cos(angle_left)
                        rx_tangent = tangent_dist * np.sin(angle_right)
                        ry_tangent = tangent_dist * np.cos(angle_right)

                        ax_map.plot([0, lx_tangent], [0, ly_tangent], color="green", alpha=0.3)
                        ax_map.plot([0, rx_tangent], [0, ry_tangent], color="green", alpha=0.3)

                    else:
                        # Listener is inside the circle
                        ax_map.plot(0, 0, marker="o", markersize=30, markeredgecolor="green", markerfacecolor="none", alpha=0.3)

                # --- 4. Timeline (Gantt) ---
                ax_time = fig.add_subplot(2, 2, 4)
                ax_time.set_title("Timeline (Relative Time [s])")

                if metadata:
                    win_start, win_end = metadata[0]["win_range"]
                    sr = 44100  # Assuming default SR

                    # Center view around window
                    # Window duration in samples
                    win_duration_samples = win_end - win_start

                    yticks = []
                    yticklabels = []

                    for i, m in enumerate(metadata):
                        y = i
                        # Convert active samples to relative time (seconds)
                        # t=0 is the start of the current window
                        start_rel_sec = (m["start_sample"] - win_start) / sr
                        end_rel_sec = (m["end_sample"] - win_start) / sr
                        duration_sec = end_rel_sec - start_rel_sec

                        ax_time.barh(y, duration_sec, left=start_rel_sec, height=0.6, align="center", color="green", alpha=0.6)
                        yticks.append(y)
                        yticklabels.append(f"Sound {m['id']}")

                    ax_time.set_yticks(yticks)
                    ax_time.set_yticklabels(yticklabels)

                    # Highlight current window [0, win_duration_in_sec]
                    win_duration_sec = win_duration_samples / sr
                    ax_time.axvspan(0, win_duration_sec, color="red", alpha=0.1, label="Current Window")
                    ax_time.axvline(win_duration_sec, color="red", linestyle="--", label="Instant")

                    # Set x-limit to show window + context
                    # Show a bit of context before and after
                    context = 1.0  # 1 second context
                    ax_time.set_xlim(-context, win_duration_sec + context)
                    ax_time.set_xlabel("Time (s) [0 = Window Start]")

                fig.suptitle(f"Epoch {epoch} — Sample", fontsize=15)
                fig.tight_layout()
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Viewer exception: {e}")

            plt.pause(0.01)


class LivePredictionPlot:
    """Persistent matplotlib window that updates only the Predictions for real-time eval."""

    def __init__(self):
        self._q = mp.Queue(maxsize=1)
        self._p = mp.Process(target=self._run_viewer, args=(self._q,), daemon=True)
        self._p.start()

    def stop(self):
        try:
            self._q.put(None, timeout=1)
        except:
            pass
        self._p.join(timeout=2.0)

    def update(self, pred_logits, current_time_sec):
        if self._q.full():
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
        try:
            self._q.put((pred_logits, current_time_sec), timeout=0.1)
        except queue.Full:
            pass

    @staticmethod
    def _run_viewer(q):
        import matplotlib

        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        plt.ion()
        fig = plt.figure(figsize=(8, 8))
        ax_pred = fig.add_subplot(1, 1, 1, projection="polar")
        mesh = None
        title_text = fig.suptitle("Waiting for first audio chunk…", fontsize=15)

        fig.canvas.draw()
        plt.pause(0.01)

        while True:
            try:
                data = q.get(timeout=0.02)  # Faster polling
                if data is None:
                    break
                pred_logits, current_time_sec = data

                pred_prob = 1.0 / (1.0 + np.exp(-pred_logits))

                if mesh is None:
                    azi_bins = pred_prob.shape[0]
                    # Create theta and r grids for pcolormesh
                    # theta includes the endpoint to close the circle
                    theta = np.linspace(0, 2 * np.pi, azi_bins + 1)
                    r = np.array([0.0, 1.0])  # Fill the center

                    # Initial data for the mesh (must be 2D)
                    mesh_data = pred_prob.reshape(1, -1)

                    mesh = ax_pred.pcolormesh(theta, r, mesh_data, cmap="magma", shading="flat", vmin=0, vmax=1)

                    ax_pred.set_ylim(0, 1)
                    ax_pred.set_theta_zero_location("N")
                    ax_pred.set_theta_direction(-1)
                    ax_pred.set_title("Predicted Location", fontsize=15, pad=20)
                    # Cleanup axis
                    ax_pred.set_rticks([])
                    ax_pred.grid(True, alpha=0.2)

                    # Add colorbar
                    cbar = fig.colorbar(mesh, ax=ax_pred, orientation="vertical", fraction=0.046, pad=0.1)
                    cbar.set_label("Probability", rotation=270, labelpad=15)

                    fig.canvas.draw()
                else:
                    # Update mesh data directly - ultra fast
                    mesh.set_array(pred_prob.flatten())

                title_text.set_text(f"Time: {current_time_sec:.1f}s")
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Viewer exception: {e}")

            plt.pause(0.001)
