from flask import Flask, render_template, request, jsonify, send_file, url_for, Response
import mimetypes

# Test OpenCV import with better error handling
try:
    import cv2
    print(f"✅ OpenCV imported successfully! Version: {cv2.__version__}")
except ImportError as e:
    print(f"❌ OpenCV import failed: {e}")
    print("💡 Try: pip uninstall opencv-python opencv-python-headless")
    print("💡 Then: pip install opencv-python-headless==4.8.1.78")
    raise

import numpy as np
import os
import json
import torch
from ultralytics import YOLO
import tempfile
from werkzeug.utils import secure_filename
from datetime import datetime
import logging

# Try to import Detectron2 detector
try:
    from detectron2_detector import Detectron2GoalLineDetector
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Ensure upload and results directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedGoalLineDetector:
    def __init__(self):
        """Initialize the advanced goal line detector with perspective-aware goal plane detection"""
        self.model = None
        self.goal_plane = None  # Will store goal plane parameters: ax + by + cz = d
        self.left_post_vector = None
        self.right_post_vector = None
        self.goal_line_points = None  # Points on the goal line
        self.frame_height = None
        self.frame_width = None
        
        # Load the custom model
        model_path = '../goal-seg/exp3050ti2/weights/best.pt'
        if os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
                logger.info(f"✅ Loaded custom trained YOLO model from {model_path}")
                if hasattr(self.model, 'names'):
                    logger.info(f"🏷️ Model class names: {self.model.names}")
                else:
                    logger.warning("⚠️ Model has no class names defined")
            except Exception as e:
                logger.error(f"❌ Error loading custom YOLO model: {e}")
                raise ValueError(f"Failed to load custom model: {e}")
        else:
            logger.error(f"❌ Custom model not found at {model_path}")
            raise ValueError(f"Custom model not found at {model_path}")
    
    def find_class_indices(self):
        """Find the class indices for ball, post, line, and crossbar in the model"""
        if not hasattr(self.model, 'names'):
            return None, None, None, None
        
        ball_idx = post_idx = line_idx = crossbar_idx = None
        for idx, name in self.model.names.items():
            name_lower = name.lower()
            if 'ball' in name_lower:
                ball_idx = idx
            elif 'post' in name_lower:
                post_idx = idx
            elif 'line' in name_lower:
                line_idx = idx
            elif 'crossbar' in name_lower:
                crossbar_idx = idx
        
        return ball_idx, post_idx, line_idx, crossbar_idx
    
    def extract_mask_info(self, mask, class_name):
        """Extract detailed information from a segmentation mask"""
        if mask is None:
            return None
        
        # Convert mask to numpy array if needed
        if hasattr(mask, 'data'):
            mask_array = mask.data.cpu().numpy().astype(np.uint8)
        else:
            mask_array = mask.astype(np.uint8)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get the largest contour (main object)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Calculate various geometric properties
        area = cv2.contourArea(main_contour)
        if area < 10:  # Too small
            return None
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # Moments for centroid
        M = cv2.moments(main_contour)
        if M["m00"] != 0:
            center_x = M["m10"] / M["m00"]
            center_y = M["m01"] / M["m00"]
        else:
            center_x, center_y = x + w/2, y + h/2
        
        # For different object types, extract specific features
        if 'ball' in class_name.lower():
            # For ball: get enclosing circle for accurate radius
            (circle_x, circle_y), radius = cv2.minEnclosingCircle(main_contour)
            return {
                'type': 'ball',
                'center': [float(circle_x), float(circle_y)],
                'radius': float(radius),
                'area': float(area),
                'contour': main_contour,
                'mask': mask_array,
                'bbox': [x, y, x+w, y+h]
            }
        
        elif 'post' in class_name.lower():
            # For post: fit line to get orientation
            [vx, vy, x0, y0] = cv2.fitLine(main_contour, cv2.DIST_L2, 0, 0.01, 0.01)
            
            # Get the top and bottom points of the post
            topmost = tuple(main_contour[main_contour[:,:,1].argmin()][0])
            bottommost = tuple(main_contour[main_contour[:,:,1].argmax()][0])
            leftmost = tuple(main_contour[main_contour[:,:,0].argmin()][0])
            rightmost = tuple(main_contour[main_contour[:,:,0].argmax()][0])
            
            return {
                'type': 'post',
                'center': [float(center_x), float(center_y)],
                'top_point': topmost,
                'bottom_point': bottommost,
                'left_point': leftmost,
                'right_point': rightmost,
                'orientation_vector': [float(vx), float(vy)],
                'line_point': [float(x0), float(y0)],
                'area': float(area),
                'contour': main_contour,
                'mask': mask_array,
                'bbox': [x, y, x+w, y+h]
            }
        
        elif 'line' in class_name.lower():
            # For line: fit line and get endpoints
            [vx, vy, x0, y0] = cv2.fitLine(main_contour, cv2.DIST_L2, 0, 0.01, 0.01)
            
            # Get leftmost and rightmost points
            leftmost = tuple(main_contour[main_contour[:,:,0].argmin()][0])
            rightmost = tuple(main_contour[main_contour[:,:,0].argmax()][0])
            
            return {
                'type': 'line',
                'center': [float(center_x), float(center_y)],
                'left_endpoint': leftmost,
                'right_endpoint': rightmost,
                'orientation_vector': [float(vx), float(vy)],
                'line_point': [float(x0), float(y0)],
                'area': float(area),
                'contour': main_contour,
                'mask': mask_array,
                'bbox': [x, y, x+w, y+h]
            }
        
        elif 'crossbar' in class_name.lower():
            # For crossbar: similar to line but horizontal
            [vx, vy, x0, y0] = cv2.fitLine(main_contour, cv2.DIST_L2, 0, 0.01, 0.01)
            
            leftmost = tuple(main_contour[main_contour[:,:,0].argmin()][0])
            rightmost = tuple(main_contour[main_contour[:,:,0].argmax()][0])
            
            return {
                'type': 'crossbar',
                'center': [float(center_x), float(center_y)],
                'left_endpoint': leftmost,
                'right_endpoint': rightmost,
                'orientation_vector': [float(vx), float(vy)],
                'line_point': [float(x0), float(y0)],
                'area': float(area),
                'contour': main_contour,
                'mask': mask_array,
                'bbox': [x, y, x+w, y+h]
            }
        
        else:
            # Generic object
            return {
                'type': 'generic',
                'center': [float(center_x), float(center_y)],
                'area': float(area),
                'contour': main_contour,
                'mask': mask_array,
                'bbox': [x, y, x+w, y+h]
            }
    
    def establish_goal_plane(self, frame):
        """Find the goal line vector for simple goal detection"""
        
        # 1. Check if model is available
        if self.model is None:
            return False
        
        # 2. Get frame dimensions
        self.frame_height, self.frame_width = frame.shape[:2]
        
        # 3. Run the model
        results = self.model(frame)[0]
        
        # 4. Get class indices
        ball_idx, post_idx, line_idx, crossbar_idx = self.find_class_indices()
        
        # 5. Look for line masks only
        line_data = []
        
        # Check if we have segmentation masks
        if not (hasattr(results, 'masks') and results.masks is not None):
            logger.warning("⚠️ No segmentation masks available")
            return False
        
        logger.info("🎭 Processing segmentation masks for line detection...")
        
        # LOOP through all detected masks looking for lines
        for i, mask in enumerate(results.masks.data):
            if i >= len(results.boxes):
                continue
                
            cls = int(results.boxes.cls[i].cpu().numpy())
            conf = float(results.boxes.conf[i].cpu().numpy())
            
            # Only process line class with good confidence
            if cls == line_idx and conf > 0.3:
                # Convert the mask to a binary image
                if hasattr(mask, 'cpu'):
                    mask_array = mask.cpu().numpy().astype(np.uint8)
                else:
                    mask_array = mask.astype(np.uint8)
                
                # Find the biggest contour from the mask
                contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                
                biggest_contour = max(contours, key=cv2.contourArea)
                
                # If the area of the contour is too small → skip
                if cv2.contourArea(biggest_contour) < 50:
                    continue
                
                # Get leftmost and rightmost points
                leftmost = tuple(biggest_contour[biggest_contour[:,:,0].argmin()][0])
                rightmost = tuple(biggest_contour[biggest_contour[:,:,0].argmax()][0])
                
                # Calculate line vector
                line_vector = np.array([rightmost[0] - leftmost[0], rightmost[1] - leftmost[1]])
                line_length = np.linalg.norm(line_vector)
                
                if line_length > 0:
                    # Normalize the vector
                    normalized_vector = line_vector / line_length
                    
                    line_info = {
                        'leftmost_point': leftmost,
                        'rightmost_point': rightmost,
                        'vector': line_vector,
                        'normalized_vector': normalized_vector,
                        'length': line_length,
                        'contour': biggest_contour,
                        'confidence': conf
                    }
                    line_data.append(line_info)
        
        logger.info(f"🏗️ Detected: {len(line_data)} goal lines")
        
        # Use the longest/most confident line
        if line_data:
            # Pick the longest line
            best_line = max(line_data, key=lambda l: l['length'])
            
            # Store the goal line info
            self.goal_line_info = best_line
            
            logger.info(f"🎯 Goal line established!")
            logger.info(f"   Leftmost point: {best_line['leftmost_point']}")
            logger.info(f"   Rightmost point: {best_line['rightmost_point']}")
            logger.info(f"   Vector: {best_line['vector']}")
            logger.info(f"   Length: {best_line['length']:.1f}px")
            
            return True
        else:
            logger.warning("⚠️ No goal line detected")
            return False
    
    def detect_ball_in_frame(self, frame):
        """Detect ball position using segmentation masks for precise center and radius"""
        if self.model is None:
            return None
            
        results = self.model(frame)[0]
        ball_idx, _, _, _ = self.find_class_indices()
        
        if ball_idx is None:
            return None
        
        balls = []
        
        # Use segmentation masks for precise ball detection
        if hasattr(results, 'masks') and results.masks is not None:
            for i, mask in enumerate(results.masks.data):
                if i < len(results.boxes):
                    cls = int(results.boxes.cls[i].cpu().numpy())
                    conf = float(results.boxes.conf[i].cpu().numpy())
                    
                    if cls == ball_idx and conf > 0.3:
                        class_name = self.model.names.get(cls, 'ball')
                        ball_info = self.extract_mask_info(mask, class_name)
                        
                        if ball_info is not None:
                            ball_info['confidence'] = conf
                            balls.append(ball_info)
        
        else:
            # Fallback to bounding boxes
            logger.warning("⚠️ No masks available for ball detection, using bounding boxes")
            if results.boxes is not None:
                for box in results.boxes:
                    cls = int(box.cls.cpu().numpy()[0])
                    conf = float(box.conf.cpu().numpy()[0])
                    
                    if cls == ball_idx and conf > 0.3:
                        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                        ball_center_x = (x1 + x2) / 2
                        ball_center_y = (y1 + y2) / 2
                        ball_radius = max((x2 - x1), (y2 - y1)) / 2
                        
                        balls.append({
                            'type': 'ball',
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'center': [ball_center_x, ball_center_y],
                            'radius': ball_radius,
                            'confidence': conf
                        })
        
        return max(balls, key=lambda b: b['confidence']) if balls else None
    
    def point_to_line_distance(self, point, line_start, line_end):
        """Calculate the perpendicular distance from a point to a line segment"""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Calculate the distance using the formula for point-to-line distance
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        
        distance = abs(A * x0 + B * y0 + C) / np.sqrt(A**2 + B**2)
        return distance
    
    def is_point_past_goal_line(self, point):
        """Check if a point has crossed the goal line using perspective-aware geometry"""
        if self.goal_line_points is None:
            return False
        
        x, y = point
        left_point = self.goal_line_points['left']
        right_point = self.goal_line_points['right']
        
        # Interpolate the goal line Y at the ball's X position
        if right_point[0] - left_point[0] == 0:  # Vertical line (shouldn't happen)
            goal_y_at_x = left_point[1]
        else:
            # Linear interpolation
            t = (x - left_point[0]) / (right_point[0] - left_point[0])
            t = max(0, min(1, t))  # Clamp between 0 and 1
            goal_y_at_x = left_point[1] + t * (right_point[1] - left_point[1])
        
        # Check if the point is past the goal line (in the goal)
        # Assuming the goal is "below" the line in image coordinates
        return y > goal_y_at_x
    
    def is_goal_event(self, ball_data):
        """Improved goal detection: check if ball is ENTIRELY crossed to the left of the goal line vector"""
        if ball_data is None or not hasattr(self, 'goal_line_info') or self.goal_line_info is None:
            return False
        
        ball_center = ball_data['center']
        ball_radius = ball_data.get('radius', 0)
        
        # Get the goal line vector and leftmost point
        leftmost_point = self.goal_line_info['leftmost_point']
        line_vector = self.goal_line_info['vector']
        
        # Normalize the line vector
        line_length = np.linalg.norm(line_vector)
        if line_length == 0:
            return False
        
        normalized_line_vector = line_vector / line_length
        
        # Get the perpendicular vector pointing to the right of the line
        # (swap x and y, negate one to get perpendicular)
        perpendicular_vector = np.array([normalized_line_vector[1], -normalized_line_vector[0]])
        
        # Calculate the rightmost point of the ball (center + radius in perpendicular direction)
        rightmost_ball_point = np.array(ball_center) + ball_radius * perpendicular_vector
        
        # Vector from leftmost point to the rightmost point of the ball
        ball_edge_vector = rightmost_ball_point - np.array(leftmost_point)
        
        # Calculate cross product to determine which side of the line the ball's rightmost edge is on
        # If cross product is positive, ball's rightmost edge is to the left of the vector (entire ball crossed = goal!)
        cross_product = np.cross(line_vector, ball_edge_vector)
        
        is_goal = cross_product > 0
        
        if is_goal:
            logger.info(f"🥅 GOAL! Ball is ENTIRELY crossed to the left of the goal line vector!")
            logger.info(f"   Ball center: ({ball_center[0]:.1f}, {ball_center[1]:.1f})")
            logger.info(f"   Ball radius: {ball_radius:.1f}px")
            logger.info(f"   Rightmost ball point: ({rightmost_ball_point[0]:.1f}, {rightmost_ball_point[1]:.1f})")
            logger.info(f"   Goal line leftmost: {leftmost_point}")
            logger.info(f"   Line vector: {line_vector}")
            logger.info(f"   Cross product: {cross_product:.1f}")
        else:
            logger.debug(f"No goal - Ball center: ({ball_center[0]:.1f}, {ball_center[1]:.1f}), "
                        f"rightmost edge: ({rightmost_ball_point[0]:.1f}, {rightmost_ball_point[1]:.1f}), "
                        f"cross product: {cross_product:.1f}")
        
        return is_goal
    
    def draw_goal_detection_overlay(self, frame):
        """Draw the simple goal line vector visualization"""
        if not hasattr(self, 'goal_line_info') or self.goal_line_info is None:
            return frame
        
        overlay = frame.copy()
        
        # Draw the goal line
        leftmost = self.goal_line_info['leftmost_point']
        rightmost = self.goal_line_info['rightmost_point']
        
        cv2.line(overlay, leftmost, rightmost, (0, 255, 255), 4)  # Yellow goal line
        
        # Draw the detection vector ALONG the goal line, same length as goal line
        # The vector should go from leftmost to rightmost (along the line)
        cv2.arrowedLine(overlay, leftmost, rightmost, (255, 0, 255), 3, tipLength=0.05)  # Magenta arrow along goal line
        
        # Mark the leftmost point (detection reference point)
        cv2.circle(overlay, leftmost, 8, (0, 255, 0), -1)  # Green circle
        
        # Mark the rightmost point 
        cv2.circle(overlay, rightmost, 6, (0, 0, 255), -1)  # Red circle
        
        # Add text
        cv2.putText(overlay, 'GOAL LINE VECTOR', 
                   (leftmost[0], leftmost[1] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(overlay, 'Goal if ball is LEFT of this vector', 
                   (leftmost[0], leftmost[1] - 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return overlay

    def process_video(self, video_path, output_path, slow_mo_factor=8):
        """Process video with simple goal line vector detection"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        # Get video properties
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        video_duration = total_frames / original_fps if original_fps > 0 else 0
        video_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        
        logger.info("🎬 ===== SIMPLE GOAL DETECTION STARTED =====")
        logger.info(f"📁 Input: {os.path.basename(video_path)} ({video_size_mb:.1f}MB)")
        logger.info(f"📐 Resolution: {width}x{height}, Duration: {video_duration:.1f}s, FPS: {original_fps}")
        logger.info(f"🎬 Slow motion factor: {slow_mo_factor}x for goal frames")
        
        # Handle rotation and output setup
        needs_rotation = False
        if width == 1080 and height == 1920:
            output_width, output_height = width, height
        elif width > height:
            needs_rotation = True
            output_width, output_height = height, width
            logger.info(f"📱 Will rotate to portrait ({output_width}x{output_height})")
        else:
            output_width, output_height = width, height
        
        # Frame sampling - keep original FPS to avoid upsampling
        TARGET_FPS = original_fps  # Use original FPS instead of forcing 60
        frame_skip = 1  # Process every frame to maintain quality
        out_fps = original_fps  # Keep original frame rate
        
        # Use H.264 codec for much better compression
        fourcc = cv2.VideoWriter_fourcc(*'H264')  # More efficient than mp4v
        out = cv2.VideoWriter(output_path, fourcc, out_fps, (output_width, output_height))
        
        # If H264 fails, try alternative codecs
        if not out.isOpened():
            logger.warning("H264 codec failed, trying mp4v...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, out_fps, (output_width, output_height))
        
        if not out.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open output for writing: {output_path}")
        
        goal_events = []
        frame_idx = 0
        processed_frames = 0
        start_time = datetime.now()
        
        # Slow motion settings for goal frames
        SLOW_MO_FACTOR = slow_mo_factor  # Use the passed slow_mo_factor
        
        # Establish goal line from multiple sample frames
        logger.info("🎯 Establishing goal line vector...")
        sample_indices = [total_frames // 4, total_frames // 2, 3 * total_frames // 4]
        
        goal_line_established = False
        for sample_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, sample_idx)
            ret, sample_frame = cap.read()
            if ret:
                if needs_rotation:
                    sample_frame = cv2.rotate(sample_frame, cv2.ROTATE_90_CLOCKWISE)
                
                if self.establish_goal_plane(sample_frame):
                    goal_line_established = True
                    break
        
        if not goal_line_established:
            logger.warning("⚠️ Could not establish goal line from samples")
            cap.release()
            out.release()
            return []
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        
        logger.info("🚀 Starting frame processing with simple vector detection...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_idx % frame_skip) == 0:
                if needs_rotation:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                # 1. Equalize contrast once
                yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
                yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
                equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

                # 2. Run your segmentation/detection model on equalized frame
                results = self.model(equalized)[0]

                # 3. If you have a method to estimate the goal line (once per scene or per frame):
                #    You can compute `self.goal_line_info` here using segmentation masks or bboxes
                #    E.g., `self.goal_line_info = self.compute_goal_line_from_masks(results)`

                # 4. Ball detection
                ball_data = self.detect_ball_in_frame(equalized)
                is_goal = False

                if ball_data is not None and hasattr(self, "goal_line_info") and self.goal_line_info:
                    # 5. Simple "ball-left-of-line" check:
                    #    – Grab the ball's center‐x
                    ball_center_x = float(ball_data["center"][0])

                    #    – Grab the line's x (e.g. leftmost pixel of the mask or bounding‐box x)
                    line_left_x = float(self.goal_line_info["leftmost_point"][0])

                    #    – If ball's x is strictly less than line_left_x, it's a goal
                    if ball_center_x < line_left_x:
                        is_goal = True

                # 6. If you don't have a precomputed goal_line_info, fallback to bboxes:
                #    – Loop through results.boxes, find the mask or bbox for "line"
                #    – Compute line_left_x once here, then apply the same check as above

                if is_goal:
                    goal_event = {
                        "frame": int(processed_frames),
                        "timestamp": float(processed_frames / out_fps),
                        "original_frame": int(frame_idx),
                        "ball_position": [int(x) for x in ball_data["bbox"]],
                        "ball_center": [float(x) for x in ball_data["center"]],
                        "ball_radius": float(ball_data["radius"]),
                        "confidence": float(ball_data["confidence"])
                    }

                    if hasattr(self, "goal_line_info") and self.goal_line_info:
                        leftmost = self.goal_line_info["leftmost_point"]
                        vector = self.goal_line_info.get("vector", (0, 0))
                        goal_event["goal_line_leftmost"] = [int(leftmost[0]), int(leftmost[1])]
                        goal_event["goal_line_vector"] = [float(vector[0]), float(vector[1])]
                    else:
                        # No line computed → you can either skip or fill with None
                        goal_event["goal_line_leftmost"] = None
                        goal_event["goal_line_vector"] = None

                    goal_events.append(goal_event)

                # 7. Draw annotations:
                #    – If you have goal_line_info, draw the line and maybe the vector
                #    – Always draw ball if detected:
                annotated = results.plot()

                
                # Add goal line vector overlay
                annotated = self.draw_goal_detection_overlay(annotated)
                
                # Add goal event highlighting
                if is_goal:
                    # EXAGGERATED GOAL CELEBRATION VISUALS! 🎉
                    
                    # Multiple thick green borders for dramatic effect
                    cv2.rectangle(annotated, (0, 0), (output_width, output_height), (0, 255, 0), 25)  # Outer thick border
                    cv2.rectangle(annotated, (15, 15), (output_width-15, output_height-15), (0, 255, 0), 15)  # Inner border
                    cv2.rectangle(annotated, (30, 30), (output_width-30, output_height-30), (255, 255, 255), 8)  # White accent border
                    
                    # Massive "GOAL!!!" text with outline effect
                    goal_text = 'GOAL!!!'
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 6.0  # HUGE text
                    thickness = 15    # THICK stroke
                    
                    # Get text size for centering
                    (text_width, text_height), _ = cv2.getTextSize(goal_text, font, font_scale, thickness)
                    text_x = (output_width - text_width) // 2
                    text_y = (output_height + text_height) // 2
                    
                    # Text shadow/outline effect (black outline)
                    shadow_offset = 8
                    cv2.putText(annotated, goal_text, 
                              (text_x + shadow_offset, text_y + shadow_offset), 
                              font, font_scale, (0, 0, 0), thickness + 6)
                    
                    # Main text (bright green)
                    cv2.putText(annotated, goal_text, 
                              (text_x, text_y), 
                              font, font_scale, (0, 255, 0), thickness)
                    
                    # White highlight on text
                    cv2.putText(annotated, goal_text, 
                              (text_x, text_y), 
                              font, font_scale, (255, 255, 255), thickness // 3)
                    
                    # Additional "GOOOOOAL!" text at top
                    celebration_text = 'GOOOOOAL!'
                    cv2.putText(annotated, celebration_text, 
                              (output_width // 2 - 400, 100), 
                              font, 3.0, (255, 255, 0), 8)  # Yellow celebration text
                    
                    # Add some animated-style effects
                    cv2.circle(annotated, (output_width // 4, output_height // 4), 50, (0, 255, 255), 12)  # Cyan circle
                    cv2.circle(annotated, (3 * output_width // 4, output_height // 4), 50, (255, 0, 255), 12)  # Magenta circle
                    cv2.circle(annotated, (output_width // 4, 3 * output_height // 4), 50, (255, 255, 0), 12)  # Yellow circle
                    cv2.circle(annotated, (3 * output_width // 4, 3 * output_height // 4), 50, (0, 255, 255), 12)  # Cyan circle
                
                # Frame info
                cv2.putText(annotated, f'Frame: {processed_frames}', 
                          (10, output_height - 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                if annotated.shape[:2] != (output_height, output_width):
                    annotated = cv2.resize(annotated, (output_width, output_height))
                
                # Write frame(s) - duplicate goal frames for slow motion effect
                if is_goal:
                    # Write the goal frame multiple times for slow motion
                    for _ in range(SLOW_MO_FACTOR):
                        out.write(annotated)
                        processed_frames += 1
                    logger.info(f"🎬 SLOW-MO: Goal frame duplicated {SLOW_MO_FACTOR}x at frame {frame_idx}")
                else:
                    # Write normal frame once
                    out.write(annotated)
                    processed_frames += 1
                
                # Progress logging
                if processed_frames % 90 == 0:
                    progress = (frame_idx / total_frames) * 100
                    logger.info(f"📈 Progress: {progress:.1f}%")
            
            frame_idx += 1
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        end_time = datetime.now()
        processing_duration = (end_time - start_time).total_seconds()
        
        logger.info("🏁 ===== SIMPLE GOAL DETECTION COMPLETED =====")
        logger.info(f"✅ Processed {processed_frames:,} frames")
        logger.info(f"⚡ Processing speed: {processed_frames / processing_duration:.1f} FPS")
        logger.info(f"🥅 Goal events detected: {len(goal_events)}")
        
        if goal_events:
            logger.info("🎯 Goal events summary:")
            for i, event in enumerate(goal_events, 1):
                timestamp = event['timestamp']
                confidence = event['confidence']
                logger.info(f"   Goal {i}: {timestamp:.1f}s (confidence: {confidence:.2f})")
        
        return goal_events

# Initialize detectors
yolo_detector = AdvancedGoalLineDetector()
detectron2_detector = None

if DETECTRON2_AVAILABLE:
    try:
        detectron2_detector = Detectron2GoalLineDetector()
        logger.info("Detectron2 detector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Detectron2 detector: {e}")
        detectron2_detector = None

@app.route('/')
def index():
    """Main page"""
    detector_info = {
        'detectron2_available': detectron2_detector is not None,
        'yolo_available': yolo_detector.model is not None
    }
    return render_template('index.html', detector_info=detector_info)

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Get detector preference
    use_detectron2 = request.form.get('use_detectron2', 'false').lower() == 'true'
    
    # Get slow motion factor (default to 8 if not provided)
    try:
        slow_mo_factor = int(request.form.get('slow_mo_factor', '8'))
        # Clamp between reasonable limits
        slow_mo_factor = max(1, min(50, slow_mo_factor))  # Between 1x and 50x
    except (ValueError, TypeError):
        slow_mo_factor = 8  # Default fallback
    
    if file:
        # Secure filename
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        
        # Save uploaded file
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        
        # Process video
        try:
            output_filename = f"processed_{filename}"
            output_path = os.path.join(app.config['RESULTS_FOLDER'], output_filename)
            
            # Choose detector
            if use_detectron2 and detectron2_detector is not None:
                logger.info("Using Detectron2 detector")
                goal_events = detectron2_detector.process_video(upload_path, output_path, slow_mo_factor)
                detector_used = "Detectron2"
            else:
                logger.info("Using YOLO detector")
                goal_events = yolo_detector.process_video(upload_path, output_path, slow_mo_factor)
                detector_used = "YOLO"
            
            # Save results metadata
            results_data = {
                'original_file': filename,
                'processed_file': output_filename,
                'goal_events': goal_events,
                'total_goals': len(goal_events),
                'upload_time': timestamp,
                'detector_used': detector_used
            }
            
            results_json_path = os.path.join(app.config['RESULTS_FOLDER'], f"results_{timestamp}.json")
            with open(results_json_path, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            return jsonify({
                'success': True,
                'results': results_data,
                'processed_video_url': url_for('download_result', filename=output_filename),
                'results_id': timestamp
            })
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return jsonify({'error': f'Error processing video: {str(e)}'}), 500

@app.route('/results/<filename>')
def download_result(filename):
    """Stream processed video with range request support for in-browser playbook"""
    file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return "File not found", 404
    
    # Get file info
    file_size = os.path.getsize(file_path)
    
    # Determine MIME type with better video format support
    mimetype, _ = mimetypes.guess_type(filename)
    if mimetype is None:
        # Better default MIME type detection for video files
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in ['.mp4', '.m4v']:
            mimetype = 'video/mp4'
        elif file_ext == '.webm':
            mimetype = 'video/webm'
        elif file_ext == '.avi':
            mimetype = 'video/x-msvideo'
        elif file_ext == '.mov':
            mimetype = 'video/quicktime'
        else:
            mimetype = 'video/mp4'  # Default fallback
    
    # Handle range requests for video streaming
    range_header = request.headers.get('Range', None)
    
    if range_header:
        # Parse range header (e.g., "bytes=0-1023")
        byte_start = 0
        byte_end = file_size - 1
        
        try:
            range_match = range_header.replace('bytes=', '').split('-')
            if len(range_match) == 2:
                if range_match[0].strip():
                    byte_start = int(range_match[0])
                if range_match[1].strip():
                    byte_end = int(range_match[1])
        except (ValueError, IndexError) as e:
            logger.warning(f"Invalid range header: {range_header}, error: {e}")
            # Fall through to serve entire file
            range_header = None
        
        if range_header:  # Only if range parsing was successful
            # Ensure range is valid
            byte_start = max(0, byte_start)
            byte_end = min(file_size - 1, byte_end)
            content_length = byte_end - byte_start + 1
            
            # Read the requested chunk
            def generate():
                try:
                    with open(file_path, 'rb') as f:
                        f.seek(byte_start)
                        remaining = content_length
                        while remaining:
                            chunk_size = min(8192, remaining)  # 8KB chunks
                            chunk = f.read(chunk_size)
                            if not chunk:
                                break
                            yield chunk
                            remaining -= len(chunk)
                except IOError as e:
                    logger.error(f"Error reading file {filename}: {e}")
                    yield b''
            
            # Return partial content response (HTTP 206) with CORS headers
            response = Response(
                generate(),
                status=206,
                headers={
                    'Content-Range': f'bytes {byte_start}-{byte_end}/{file_size}',
                    'Accept-Ranges': 'bytes',
                    'Content-Length': str(content_length),
                    'Content-Type': mimetype,
                    'Cache-Control': 'no-cache',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
                    'Access-Control-Allow-Headers': 'Range'
                }
            )
            
            logger.info(f"📺 Streaming range {byte_start}-{byte_end}/{file_size} for {filename}")
            return response
    
    # No range request or invalid range - serve entire file
    def generate():
        try:
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(8192)  # 8KB chunks
                    if not chunk:
                        break
                    yield chunk
        except IOError as e:
            logger.error(f"Error reading file {filename}: {e}")
            yield b''
    
    response = Response(
        generate(),
        headers={
            'Content-Length': str(file_size),
            'Content-Type': mimetype,
            'Accept-Ranges': 'bytes',
            'Cache-Control': 'no-cache',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
            'Access-Control-Allow-Headers': 'Range'
        }
    )
    
    logger.info(f"📺 Streaming entire file {filename} ({file_size} bytes)")
    return response

@app.route('/api/results/<results_id>')
def get_results(results_id):
    """Get results data"""
    results_path = os.path.join(app.config['RESULTS_FOLDER'], f"results_{results_id}.json")
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'Results not found'}), 404

@app.route('/api/detector-status')
def detector_status():
    """Get detector availability status"""
    return jsonify({
        'detectron2_available': detectron2_detector is not None,
        'yolo_available': yolo_detector.model is not None,
        'detectron2_cuda': torch.cuda.is_available() if detectron2_detector else False
    })

@app.route('/api/test-video/<filename>')
def test_video(filename):
    """Simple test route to check if video file exists and basic info"""
    file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found', 'path': file_path}), 404
    
    try:
        file_size = os.path.getsize(file_path)
        file_ext = os.path.splitext(filename)[1].lower()
        
        return jsonify({
            'exists': True,
            'filename': filename,
            'path': file_path,
            'size': file_size,
            'size_mb': round(file_size / (1024 * 1024), 2),
            'extension': file_ext,
            'download_url': url_for('download_result', filename=filename)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/list-results')
def list_results():
    """List all available result files for debugging"""
    try:
        files = []
        if os.path.exists(app.config['RESULTS_FOLDER']):
            for filename in os.listdir(app.config['RESULTS_FOLDER']):
                file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                    files.append({
                        'filename': filename,
                        'size': file_size,
                        'size_mb': round(file_size / (1024 * 1024), 2),
                        'url': url_for('download_result', filename=filename) if filename.endswith(('.mp4', '.avi', '.mov', '.webm')) else None
                    })
        
        return jsonify({
            'results_folder': app.config['RESULTS_FOLDER'],
            'files': files,
            'total_files': len(files)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 