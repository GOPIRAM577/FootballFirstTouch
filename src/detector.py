import cv2
from ultralytics import YOLO

class Detector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initializes the YOLOv8 detector.
        Downloads the model if it doesn't exist.
        """
        self.model = YOLO(model_path)
        
        # YOLOv8 COCO classes
        self.PERSON_CLASS = 0
        self.SPORTS_BALL_CLASS = 32

    def detect(self, frame, conf_threshold=0.25):
        """
        Detects players and the ball in a given frame.

        Args:
            frame: BGR image frame from OpenCV.
            conf_threshold: Minimum confidence threshold.

        Returns:
            players: List of bounding boxes [x1, y1, x2, y2, conf] for players.
            balls: List of bounding boxes [x1, y1, x2, y2, conf] for balls.
        """
        results = self.model(frame, verbose=False)[0]
        
        players = []
        balls = []
        
        for box in results.boxes:
            cls = int(box.cls[0].item())
            conf = box.conf[0].item()
            
            if conf >= conf_threshold:
                # [x1, y1, x2, y2]
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                detection = xyxy + [conf]
                
                if cls == self.PERSON_CLASS:
                    players.append(detection)
                elif cls == self.SPORTS_BALL_CLASS:
                    balls.append(detection)
                    
        return players, balls
