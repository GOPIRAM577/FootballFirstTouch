import cv2
import numpy as np
import imageio

class Visualizer:
    def __init__(self, input_path, output_path):
        self.cap = cv2.VideoCapture(input_path)
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0 or np.isnan(self.fps):
            self.fps = 30
            
        # Use imageio to write out native H264 which runs seamlessly in Streamlit
        self.out = imageio.get_writer(output_path, fps=self.fps, codec='libx264', macro_block_size=1)

    def create_video(self, tracked_ball_raw, smoothed_ball_traj, players_per_frame, analysis_results):
        frame_idx = 0
        
        touch_frame = analysis_results.get("touch_frame", -1)
        score = analysis_results.get("score", 0.0)
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # 1. Draw Players (Thick Neon Green boxes)
            if frame_idx < len(players_per_frame):
                for p in players_per_frame[frame_idx]:
                    x1, y1, x2, y2 = int(p[0]), int(p[1]), int(p[2]), int(p[3])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3) # Neon Green Box
                    
            # 2. Draw Ball (Large circled with outline so it pops on any background)
            if frame_idx < len(tracked_ball_raw) and tracked_ball_raw[frame_idx] is not None:
                bx, by = int(tracked_ball_raw[frame_idx][0]), int(tracked_ball_raw[frame_idx][1])
                cv2.circle(frame, (bx, by), 8, (255, 255, 255), -1) # Solid White Center
                cv2.circle(frame, (bx, by), 10, (0, 0, 255), 3) # Bright Red Outer Ring
                
            # 3. Draw smoothed trajectory up to current frame
            for i in range(1, frame_idx + 1):
                if i < len(smoothed_ball_traj):
                    if smoothed_ball_traj[i-1] is not None and smoothed_ball_traj[i] is not None:
                        p1 = (int(smoothed_ball_traj[i-1][0]), int(smoothed_ball_traj[i-1][1]))
                        p2 = (int(smoothed_ball_traj[i][0]), int(smoothed_ball_traj[i][1]))
                        
                        # Change color after touch
                        color = (0, 255, 255) if i <= touch_frame else (255, 0, 255) # Yellow before, Magenta after
                        cv2.line(frame, p1, p2, color, 4) # Thicker trajectory line
                        
            # 4. Highlight Touch interaction
            if frame_idx == touch_frame:
                cv2.putText(frame, "TOUCH", (self.width // 2 - 50, self.height // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                            
            # 5. Display Metrics On-Screen
            if frame_idx >= touch_frame and touch_frame != -1:
                self._draw_metrics(frame, analysis_results)
                
            # Write using imageio (requires RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.out.append_data(rgb_frame)
            frame_idx += 1
            
        self.cap.release()
        self.out.close()
        
    def _draw_metrics(self, frame, res):
        y_offset = 30
        metrics_text = [
            f"First Touch Score: {res.get('score', 0):.2f}/10",
            f"Speed Ratio: {res.get('speed_ratio', 0):.2f}",
            f"Direction Change: {res.get('direction_change', 0):.1f} deg",
            f"Control Distance: {res.get('control_distance', 0):.1f} px",
            f"Pressure Factor: {res.get('pressure_factor', 0):.2f}"
        ]
        
        # Draw semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 40 + len(metrics_text)*30), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        for text in metrics_text:
            cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30

