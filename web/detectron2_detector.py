import cv2
import numpy as np
import torch
import os
import logging
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import detectron2.data.transforms as T

logger = logging.getLogger(__name__)

class Detectron2GoalLineDetector:
    def __init__(self):
        """Initialize the Detectron2-based goal line detector"""
        self.predictor = None
        self.class_names = {0: 'ball', 1: 'post', 2: 'line'}
        self.load_model()
    
    def load_model(self):
        """Load the trained Detectron2 model"""
        try:
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            
            # Try to load your trained model
            model_path = "../output/model_final.pth"
            if os.path.exists(model_path):
                cfg.MODEL.WEIGHTS = model_path
                logger.info(f"Loaded trained model from {model_path}")
            else:
                # Fallback to pre-trained weights
                cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
                logger.warning("Trained model not found, using pre-trained weights")
            
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # ball, post, line
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
            cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.predictor = DefaultPredictor(cfg)
            logger.info("Detectron2 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading Detectron2 model: {e}")
            self.predictor = None
    
    def detect_objects(self, frame):
        """Detect ball, posts, and lines in a frame using Detectron2"""
        if self.predictor is None:
            return [], [], []
        
        try:
            outputs = self.predictor(frame)
            
            instances = outputs["instances"]
            boxes = instances.pred_boxes.tensor.cpu().numpy()
            classes = instances.pred_classes.cpu().numpy()
            scores = instances.scores.cpu().numpy()
            masks = instances.pred_masks.cpu().numpy() if instances.has("pred_masks") else None
            
            balls = []
            posts = []
            lines = []
            
            for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
                if score > 0.5:  # Confidence threshold
                    detection = {
                        'bbox': box.astype(int),
                        'score': score,
                        'mask': masks[i] if masks is not None else None
                    }
                    
                    if cls == 0:  # ball
                        balls.append(detection)
                    elif cls == 1:  # post
                        posts.append(detection)
                    elif cls == 2:  # line
                        lines.append(detection)
            
            return balls, posts, lines
            
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            return [], [], []
    
    def extract_goal_line(self, line_detections, frame_shape):
        """Extract the most likely goal line from line detections"""
        if not line_detections:
            return None
        
        # Find the most confident line detection
        best_line = max(line_detections, key=lambda x: x['score'])
        
        if best_line['mask'] is not None:
            # Extract line from mask
            mask = best_line['mask']
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the longest contour (likely the goal line)
                longest_contour = max(contours, key=cv2.contourArea)
                
                # Fit a line to the contour
                [vx, vy, x, y] = cv2.fitLine(longest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
                
                # Calculate line endpoints
                lefty = int((-x * vy / vx) + y)
                righty = int(((frame_shape[1] - x) * vy / vx) + y)
                
                return [0, lefty, frame_shape[1] - 1, righty]
        
        # Fallback to bounding box
        bbox = best_line['bbox']
        return [bbox[0], (bbox[1] + bbox[3]) // 2, bbox[2], (bbox[1] + bbox[3]) // 2]
    
    def is_ball_crossing_line(self, ball_detections, goal_line, frame_height, prev_ball_positions=None):
        """Determine if any ball is crossing the goal line"""
        if not ball_detections or goal_line is None:
            return False, None
        
        goal_events = []
        
        for ball in ball_detections:
            bbox = ball['bbox']
            ball_center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
            
            # Calculate distance from ball center to goal line
            line_y = (goal_line[1] + goal_line[3]) // 2
            distance_to_line = abs(ball_center[1] - line_y)
            
            # Check if ball is close to the line (within 5% of frame height)
            threshold = frame_height * 0.05
            
            if distance_to_line < threshold:
                # Additional check: ball should be crossing from above to below (or vice versa)
                crossing_detected = True
                
                # If we have previous positions, check for actual crossing
                if prev_ball_positions:
                    # This would require implementing ball tracking across frames
                    # For now, we'll use proximity as the main indicator
                    pass
                
                if crossing_detected:
                    goal_events.append({
                        'ball_bbox': bbox,
                        'ball_center': ball_center,
                        'distance_to_line': distance_to_line,
                        'confidence': ball['score']
                    })
        
        return len(goal_events) > 0, goal_events
    
    def process_frame(self, frame, frame_number, fps):
        """Process a single frame and return detections and goal events"""
        # Detect objects
        balls, posts, lines = self.detect_objects(frame)
        
        # Extract goal line
        goal_line = self.extract_goal_line(lines, frame.shape)
        
        # Check for goal events
        is_goal, goal_events = self.is_ball_crossing_line(balls, goal_line, frame.shape[0])
        
        # Prepare results
        results = {
            'balls': balls,
            'posts': posts,
            'lines': lines,
            'goal_line': goal_line,
            'is_goal': is_goal,
            'goal_events': goal_events,
            'frame_number': frame_number,
            'timestamp': frame_number / fps
        }
        
        return results
    
    def annotate_frame(self, frame, detection_results):
        """Annotate frame with detections and goal events"""
        annotated_frame = frame.copy()
        
        # Draw ball detections
        for ball in detection_results['balls']:
            bbox = ball['bbox']
            cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Ball {ball['score']:.2f}", 
                      (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw post detections
        for post in detection_results['posts']:
            bbox = post['bbox']
            cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)
            cv2.putText(annotated_frame, f"Post {post['score']:.2f}", 
                      (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Draw line detections
        for line in detection_results['lines']:
            if line['mask'] is not None:
                # Draw mask
                mask = line['mask']
                colored_mask = np.zeros_like(frame)
                colored_mask[mask] = [255, 0, 0]
                annotated_frame = cv2.addWeighted(annotated_frame, 0.8, colored_mask, 0.2, 0)
            else:
                bbox = line['bbox']
                cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        
        # Draw goal line
        if detection_results['goal_line'] is not None:
            goal_line = detection_results['goal_line']
            cv2.line(annotated_frame, (goal_line[0], goal_line[1]), 
                    (goal_line[2], goal_line[3]), (0, 0, 255), 3)
            cv2.putText(annotated_frame, 'Goal Line', 
                      (goal_line[0], goal_line[1] - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Highlight goal events
        if detection_results['is_goal']:
            cv2.putText(annotated_frame, 'GOAL!', 
                      (frame.shape[1] // 2 - 100, 100), 
                      cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            cv2.rectangle(annotated_frame, (0, 0), 
                        (frame.shape[1], frame.shape[0]), (0, 0, 255), 8)
        
        # Add frame info
        cv2.putText(annotated_frame, f'Frame: {detection_results["frame_number"]}', 
                  (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f'Time: {detection_results["timestamp"]:.2f}s', 
                  (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated_frame
    
    def process_video(self, video_path, output_path, slow_mo_factor=8):
        """Process entire video and detect goal events"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_goal_events = []
        frame_number = 0
        
        logger.info(f"Processing video with Detectron2: {total_frames} frames at {fps} FPS")
        logger.info(f"🎬 Slow motion factor: {slow_mo_factor}x for goal frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            detection_results = self.process_frame(frame, frame_number, fps)
            
            # Collect goal events
            if detection_results['is_goal']:
                for event in detection_results['goal_events']:
                    all_goal_events.append({
                        'frame': frame_number,
                        'timestamp': detection_results['timestamp'],
                        'ball_position': event['ball_bbox'].tolist(),
                        'ball_center': event['ball_center'],
                        'confidence': event['confidence'],
                        'distance_to_line': event['distance_to_line']
                    })
            
            # Annotate frame
            annotated_frame = self.annotate_frame(frame, detection_results)
            
            # Write frame(s) - duplicate goal frames for slow motion effect
            if detection_results['is_goal']:
                # Write the goal frame multiple times for slow motion
                for _ in range(slow_mo_factor):
                    out.write(annotated_frame)
                logger.info(f"🎬 SLOW-MO: Goal frame duplicated {slow_mo_factor}x at frame {frame_number}")
            else:
                # Write normal frame once
                out.write(annotated_frame)
            
            frame_number += 1
            
            # Progress logging
            if frame_number % 100 == 0:
                progress = (frame_number / total_frames) * 100
                logger.info(f"Detectron2 processing progress: {progress:.1f}%")
        
        cap.release()
        out.release()
        
        return all_goal_events 