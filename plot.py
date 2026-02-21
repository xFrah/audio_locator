import multiprocessing as mp
import queue
import numpy as np

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
        pass # Deprecated, handled by background process

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
                print(f"Viewer exception: {e}")
            
            plt.pause(0.01)
