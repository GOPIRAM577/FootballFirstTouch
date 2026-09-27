import numpy as np
import math

class TouchAnalyzer:
    def __init__(self, fps=30):
        self.fps = fps
        # Hyperparameters
        self.pre_frames = 5
        self.post_frames = 15
        
    def analyze_touch(self, ball_traj, players_per_frame):
        """
        Analyzes the first touch scenario given a full trajectory.
        
        Args:
            ball_traj: List of (x, y) tuples representing the ball's center.
            players_per_frame: List of Lists of [x1, y1, x2, y2, conf] for players per frame.
            
        Returns:
            metrics: dict containing touch details and score
        """
        num_frames = len(ball_traj)
        if num_frames == 0 or len(players_per_frame) == 0:
            return {"error": "Insufficient data"}
            
        # 1. Identify the touch frame and receiving player
        # We assume the touch happens when the ball is nearest to a player.
        min_dist_overall = float('inf')
        touch_frame = -1
        receiver_idx = -1 # index of the receiver in that specific frame
        
        for i in range(num_frames):
            b_pos = ball_traj[i]
            players = players_per_frame[i]
            
            if b_pos is None or len(players) == 0:
                continue
                
            b_x, b_y = b_pos
            
            for j, p in enumerate(players):
                # Calculate distance between ball and player center
                px = (p[0] + p[2]) / 2
                py = (p[1] + p[3]) / 2
                dist = math.dist((b_x, b_y), (px, py))
                
                # We could also check if ball intersects player bounding box
                
                if dist < min_dist_overall:
                    min_dist_overall = dist
                    touch_frame = i
                    receiver_idx = j
                    
        if touch_frame == -1 or touch_frame < self.pre_frames or touch_frame > num_frames - self.post_frames:
            # Check if there is ANY valid ball tracking in the video
            valid_ball_frames = sum([1 for pos in ball_traj if pos is not None])
            if valid_ball_frames == 0:
                # Genuinely no ball detected anywhere — completely empty video
                return {
                    "error": "no_touch",
                    "touch_frame": -1,
                    "speed_before": 0, "speed_after": 0, "speed_ratio": 0,
                    "direction_change": 0, "control_distance": 0,
                    "pressure_factor": 0, "score": 0.0,
                    "classification_label": "No Ball Detected (Ensure lighting is good)"
                }
            
            # Fallback to middle of the detected ball sequence if a specific touch wasn't perfectly framed
            # We enforce limits to prevent crashing
            touch_frame = max(self.pre_frames, min(num_frames - self.post_frames - 1, num_frames // 2))
            
            if len(players_per_frame) > touch_frame and len(players_per_frame[touch_frame]) > 0:
                receiver_idx = 0
            else:
                receiver_idx = -1
        # The receiver is the player at receiver_idx in the touch_frame
        # To identify them in other frames, we can use simple distance tracking or assume they stay somewhat in similar regions.
        receivers_per_frame = self._track_receiver(touch_frame, receiver_idx, players_per_frame)

        # 2. Extract metrics (Speed, Direction, Distance Control)
        speed_before = self._calculate_speed(ball_traj, max(0, touch_frame - self.pre_frames), touch_frame)
        speed_after = self._calculate_speed(ball_traj, touch_frame, min(num_frames - 1, touch_frame + self.post_frames))
        
        speed_ratio = speed_after / (speed_before + 1e-5) # Prevent div by 0
        
        # Angle deviation
        angle_before = self._calculate_direction(ball_traj, max(0, touch_frame - self.pre_frames), touch_frame)
        angle_after = self._calculate_direction(ball_traj, touch_frame, min(num_frames - 1, touch_frame + self.post_frames))
        direction_change = abs(angle_before - angle_after)
        if direction_change > 180:
            direction_change = 360 - direction_change
            
        # Control Distance (Distance from receiver after touch)
        eval_frame = min(num_frames - 1, touch_frame + self.post_frames)
        r_pos_eval = receivers_per_frame[eval_frame]
        b_pos_eval = ball_traj[eval_frame]
        
        control_distance = 0
        if r_pos_eval is not None and b_pos_eval is not None:
            control_distance = math.dist(b_pos_eval, r_pos_eval)
        
        # 3. Context Awareness (Pressure)
        pressure_factor = self._calculate_pressure(touch_frame, receivers_per_frame[touch_frame], players_per_frame[touch_frame])
        
        # 4. Score Calculation — detect no real activity
        if speed_ratio == 0 and direction_change == 0 and control_distance == 0:
            score = 0.0
            classification_label = "No Touch Detected"
        else:
            score = self._compute_score(speed_ratio, direction_change, control_distance, pressure_factor)
            classification_label = None
        
        result = {
            "touch_frame": touch_frame,
            "speed_before": speed_before,
            "speed_after": speed_after,
            "speed_ratio": speed_ratio,
            "direction_change": direction_change,
            "control_distance": control_distance,
            "pressure_factor": pressure_factor,
            "score": score
        }
        
        if classification_label:
            result['classification_label'] = classification_label
            
        return result

    def _track_receiver(self, touch_frame, receiver_idx, players_per_frame):
        # A simple method to find the receiver in each frame by finding the player closest to their position in the previous/next frame
        receivers = [None] * len(players_per_frame)
        
        # Touch frame position
        receivers = [None] * len(players_per_frame)
        if receiver_idx != -1 and len(players_per_frame[touch_frame]) > 0:
            p_touch = players_per_frame[touch_frame][receiver_idx]
            rx = (p_touch[0] + p_touch[2]) / 2
            ry = (p_touch[1] + p_touch[3]) / 2
            receivers[touch_frame] = (rx, ry)
        else:
            # Fake player pos in center if literally no one is found
            rx, ry = 300, 300
            receivers[touch_frame] = (rx, ry)
            
        # Track forwards
        curr_pos = (rx, ry)
        for i in range(touch_frame + 1, len(players_per_frame)):
            if len(players_per_frame[i]) > 0:
                best_j = -1
                best_dist = float('inf')
                for j, p in enumerate(players_per_frame[i]):
                    px = (p[0] + p[2]) / 2
                    py = (p[1] + p[3]) / 2
                    d = math.dist((px, py), curr_pos)
                    if d < best_dist:
                        best_dist = d
                        best_j = j
                
                # Only update if the distance is reasonable (they didn't jump across screen)
                if best_dist < 100: 
                    p = players_per_frame[i][best_j]
                    curr_pos = ((p[0] + p[2]) / 2, (p[1] + p[3]) / 2)
                    receivers[i] = curr_pos
                    
        # Track backwards
        curr_pos = (rx, ry)
        for i in range(touch_frame - 1, -1, -1):
            if len(players_per_frame[i]) > 0:
                best_j = -1
                best_dist = float('inf')
                for j, p in enumerate(players_per_frame[i]):
                    px = (p[0] + p[2]) / 2
                    py = (p[1] + p[3]) / 2
                    d = math.dist((px, py), curr_pos)
                    if d < best_dist:
                        best_dist = d
                        best_j = j
                
                if best_dist < 100: 
                    p = players_per_frame[i][best_j]
                    curr_pos = ((p[0] + p[2]) / 2, (p[1] + p[3]) / 2)
                    receivers[i] = curr_pos
                    
        return receivers

    def _calculate_speed(self, traj, start_f, end_f):
        # Calculates average pixel distance per frame
        dist = 0
        valid_frames = 0
        for i in range(start_f, end_f - 1):
            p1 = traj[i]
            p2 = traj[i+1]
            if p1 is not None and p2 is not None:
                dist += math.dist(p1, p2)
                valid_frames += 1
                
        if valid_frames == 0:
            return 0
        return dist / valid_frames
        
    def _calculate_direction(self, traj, start_f, end_f):
        # Calculate angle of movement vector
        # Find first valid and last valid
        p_start = None
        p_end = None
        
        for i in range(start_f, end_f):
            if traj[i] is not None:
                p_start = traj[i]
                break
                
        for i in range(end_f - 1, start_f - 1, -1):
            if traj[i] is not None:
                p_end = traj[i]
                break
                
        if p_start is None or p_end is None or p_start == p_end:
            return 0
            
        dx = p_end[0] - p_start[0]
        dy = p_end[1] - p_start[1]
        
        angle = math.degrees(math.atan2(dy, dx))
        return angle

    def _calculate_pressure(self, touch_frame, receiver_pos, all_players):
        """
        Calculates pressure based on nearest defender distance.
        Returns a factor (0 to 1) where 1 means high pressure.
        """
        if receiver_pos is None or len(all_players) <= 1:
            return 0.0 # No pressure
            
        min_defender_dist = float('inf')
        for p in all_players:
            px = (p[0] + p[2]) / 2
            py = (p[1] + p[3]) / 2
            
            # Skip if this is the receiver
            if math.dist((px, py), receiver_pos) < 20: 
                continue
                
            dist = math.dist((px, py), receiver_pos)
            if dist < min_defender_dist:
                min_defender_dist = dist
                
        # Empirically scaled: if defender is < 50 pixels away, max pressure (1.0)
        # If > 300 pixels away, no pressure (0.0)
        pressure = max(0.0, min(1.0, 1.0 - (min_defender_dist - 50) / 250))
        return pressure

    def _compute_score(self, speed_ratio, direction_change, control_distance, pressure):
        """
        Combines metrics into a score out of 10.
        """
        # Baseline score
        score = 8.0 
        
        # Penalties
        # 1. High speed ratio (ball bounced away fast) -> bad control
        if speed_ratio > 0.8:
            score -= min(3.0, (speed_ratio - 0.8) * 5)
            
        # 2. Control distance (ball too far from player) -> bad control
        if control_distance > 100:
            score -= min(4.0, (control_distance - 100) / 50)
            
        # 3. Direction change (ball bounced off unpredictably) -> can be good or bad depending on context
        # For a standard control touch, we want some predictability. High deviation > 90 deg might mean lost ball.
        if direction_change > 90:
            score -= 1.0
            
        # Bonus for pressure
        # Controlling well under pressure adds points
        if pressure > 0.3 and score >= 6.0:
            score += pressure * 1.5
            
        return max(0.1, min(10.0, score))

