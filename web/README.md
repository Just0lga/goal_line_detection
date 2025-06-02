# Goal Line Detection Web Application

A sophisticated web application that uses AI computer vision to detect when a ball crosses the goal line in football/soccer videos. The system provides frame-level precision and generates annotated videos with detected goal events.

## Features

🎯 **Accurate Detection**: Uses advanced AI models (YOLO/Detectron2) to detect balls, goal posts, and goal lines
📹 **Video Processing**: Processes entire videos frame-by-frame with real-time annotations
🎨 **Modern UI**: Beautiful, responsive web interface with drag-and-drop video upload
📊 **Detailed Results**: Frame-by-frame analysis with timestamps and confidence scores
💾 **Download Results**: Get annotated videos and detailed JSON reports
🚀 **Dual AI Support**: Choose between YOLO (fast) or Detectron2 (accurate) detection

## Architecture

The application consists of several key components:

### 1. Backend (Flask)
- **app.py**: Main Flask application with video upload and processing endpoints
- **detectron2_detector.py**: Advanced detector using your trained Detectron2 model
- **GoalLineDetector class**: YOLO-based detector with Hough line detection

### 2. Frontend
- **templates/index.html**: Modern, responsive web interface
- Real-time progress tracking
- Interactive results display
- Video preview and download

### 3. AI Models
- **YOLO Detection**: Fast object detection for balls
- **Detectron2 Detection**: Instance segmentation for balls, posts, and lines
- **OpenCV Processing**: Line detection and geometric analysis

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)
- Virtual environment (recommended)

### 1. Set Up Environment
```bash
cd web
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Detectron2 (Optional but Recommended)
```bash
# For CUDA support
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html

# For CPU only
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html
```

## Usage

### 1. Start the Application
```bash
python app.py
```

The application will be available at `http://localhost:5000`

### 2. Upload and Process Videos
1. Open your web browser and navigate to `http://localhost:5000`
2. Drag and drop a video file or click to browse
3. Choose your preferred AI detector (Detectron2 or YOLO)
4. Wait for processing to complete
5. View results and download the annotated video

### 3. Supported Video Formats
- MP4, AVI, MOV, MKV
- Maximum file size: 100MB
- Recommended resolution: 720p-1080p

## API Endpoints

### POST /upload
Upload and process a video file.

**Parameters:**
- `video`: Video file (multipart/form-data)
- `use_detectron2`: Boolean (optional, default: false)

**Response:**
```json
{
  "success": true,
  "results": {
    "original_file": "video.mp4",
    "processed_file": "processed_video.mp4",
    "goal_events": [...],
    "total_goals": 2,
    "detector_used": "Detectron2"
  },
  "processed_video_url": "/results/processed_video.mp4"
}
```

### GET /results/{filename}
Download a processed video file.

### GET /api/results/{results_id}
Get detailed results data in JSON format.

### GET /api/detector-status
Check availability of AI detectors.

## Detection Algorithm

### YOLO Detector
1. **Ball Detection**: Uses YOLO model to detect soccer balls
2. **Line Detection**: Canny edge detection + Hough transform for goal line
3. **Goal Event**: Triggered when ball is within 5% of frame height from goal line

### Detectron2 Detector (Advanced)
1. **Object Detection**: Detects balls, goal posts, and lines using instance segmentation
2. **Line Extraction**: Extracts precise goal line from segmentation masks
3. **Spatial Analysis**: Advanced geometric analysis for accurate goal detection
4. **Confidence Scoring**: Multi-factor confidence calculation

## Configuration

### Model Paths
- YOLO model (primary): `../goal-seg/exp3050ti2/weights/best.pt` (your custom trained model)
- YOLO model (fallback): `../yolo11n.pt` (alternative model)
- Detectron2 model: `../output/model_final.pth` (your trained model)

### Detection Parameters
```python
# Confidence thresholds
BALL_CONFIDENCE = 0.5
LINE_CONFIDENCE = 0.5

# Goal line proximity threshold (% of frame height)
GOAL_THRESHOLD = 0.05

# Line detection parameters
CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_THRESHOLD = 100
```

## Performance Optimization

### For Better Results:
1. **Use trained models**: Place your trained YOLO/Detectron2 models in the correct paths
2. **GPU acceleration**: Ensure CUDA is available for faster processing
3. **Video quality**: Higher resolution videos generally give better detection results
4. **Lighting conditions**: Well-lit videos with clear goal lines work best

### Processing Speed:
- **YOLO**: ~30-60 FPS (depending on hardware)
- **Detectron2**: ~5-15 FPS (more accurate but slower)

## Troubleshooting

### Common Issues:

1. **"No module named 'detectron2'"**
   - Install Detectron2 following the installation instructions above
   - The app will fall back to YOLO if Detectron2 is not available

2. **"CUDA out of memory"**
   - Reduce video resolution or use CPU processing
   - Set `cfg.MODEL.DEVICE = "cpu"` in detectron2_detector.py

3. **Poor detection accuracy**
   - Ensure you're using your trained models (not the fallback pre-trained ones)
   - Adjust confidence thresholds in the detector classes
   - Check video quality and lighting conditions

4. **Slow processing**
   - Use YOLO detector for faster processing
   - Enable GPU acceleration
   - Consider processing shorter video clips

## Development

### Project Structure
```
web/
├── app.py                  # Main Flask application
├── detectron2_detector.py  # Detectron2-based detector
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── templates/
│   └── index.html         # Web interface
├── uploads/               # Uploaded videos (created automatically)
├── results/               # Processed videos and results (created automatically)
└── static/                # Static files (if any)
```

### Adding New Features
1. **Custom detectors**: Implement the same interface as GoalLineDetector
2. **New endpoints**: Add routes in app.py
3. **UI enhancements**: Modify templates/index.html
4. **Detection algorithms**: Extend detection logic in detector classes

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of your goal line detection system. Please refer to your project's license terms.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Verify your model files are in the correct locations
3. Check the application logs for detailed error messages
4. Ensure all dependencies are correctly installed

---

**Note**: This application leverages your existing trained models. Make sure your YOLO and Detectron2 models are properly trained on your dataset for optimal performance. 