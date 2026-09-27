import numpy as np

class BallTracker:
    def __init__(self, max_missed_frames=5):
        """
        Tracker for the football that handles missing detections.
        """
        self.trajectory = [] # List of (x_center, y_center) tuples, or None
        self.max_missed_frames = max_missed_frames
        self.missed_frames_count = 0
        self.raw_trajectory = [] # Includes None for missed frames

    def update(self, balls):
        """
        Update the tracker with the latest detected balls.
        
        Args:
            balls: List of [x1, y1, x2, y2, conf]
        """
        center = None
        
        if len(balls) > 0:
            # If multiple balls detected, take the one with highest confidence
            # or the one closest to previous known position
            
            # Simple approach: highest confidence
            best_ball = max(balls, key=lambda b: b[4])
            x1, y1, x2, y2, _ = best_ball
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            self.missed_frames_count = 0
        else:
            self.missed_frames_count += 1
            
        self.raw_trajectory.append(center)
        
        # We can implement basic interpolation here for missing frames if needed
        # For now, we will wait until the end of the video to interpolate the full trajectory
        
    def get_trajectory(self):
        """
        Returns the raw trajectory filled with None for missing frames.
        """
        return self.raw_trajectory
        
    def get_smoothed_trajectory(self):
        """
        Interpolates missing frames and optionally smooths the trajectory.
        """
        # Interpolate None values using pandas or numpy
        # For simplicity, let's use a very basic linear interpolation
        traj = np.array(self.raw_trajectory, dtype=object)
        
        # Find indices of valid and invalid points
        valid_idx = np.where(traj != None)[0]
        if len(valid_idx) < 2:
            return traj # Not enough points to interpolate/track
            
        # Separate x and y valid points
        valid_x = [p[0] for p in traj[valid_idx]]
        valid_y = [p[1] for p in traj[valid_idx]]
        
        # Interpolate for all indices
        all_idx = np.arange(len(traj))
        interp_x = np.interp(all_idx, valid_idx, valid_x)
        interp_y = np.interp(all_idx, valid_idx, valid_y)
        
        # Recombine
        smoothed_traj = list(zip(interp_x, interp_y))
        return smoothed_traj

class PlayerTracker:
    # A simple tracker to associate players over frames
    # and identify the receiver receiving the ball.
    def __init__(self):
        pass
