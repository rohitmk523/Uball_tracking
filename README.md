# Basketball Player Tracking System

A comprehensive computer vision pipeline for tracking basketball players using YOLO11 + SAM2 + DeepSORT, designed for elevated camera angles and multi-player scenarios.

## 🏀 Overview

This project implements a three-stage basketball player tracking system:

1. **Detection**: YOLOv11 Nano for fast player detection
2. **Segmentation**: SAM2 for pixel-accurate player boundaries  
3. **Tracking**: DeepSORT for consistent player ID retention across frames

Perfect for basketball analytics, game highlights, and player performance analysis.

## 🚀 Features

- **Real-time Processing**: 30+ FPS on modern GPUs
- **Multi-player Support**: Track 10+ players simultaneously
- **Team Classification**: Red vs Green jersey detection
- **Cross-frame Consistency**: Maintain player IDs throughout video
- **Cloud Ready**: Designed for scalable cloud deployment
- **Basketball Optimized**: Handles occlusions, picks, and basketball-specific movements

## 📁 Project Structure

```
basketball-player-tracking/
├── README.md                     # This file
├── IMPLEMENTATION_GUIDE.md       # Step-by-step implementation guide
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore file
├── config/
│   ├── yolo_config.yaml         # YOLO model configuration
│   ├── sam2_config.yaml         # SAM2 model configuration
│   └── deepsort_config.yaml     # DeepSORT tracker configuration
├── src/
│   ├── __init__.py
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── yolo_detector.py     # YOLO11 detection wrapper
│   │   └── utils.py             # Detection utilities
│   ├── segmentation/
│   │   ├── __init__.py
│   │   ├── sam2_segmenter.py    # SAM2 segmentation wrapper
│   │   └── utils.py             # Segmentation utilities
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── deepsort_tracker.py  # DeepSORT tracking wrapper
│   │   ├── kalman_filter.py     # Kalman filter for prediction
│   │   └── utils.py             # Tracking utilities
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── basketball_tracker.py # Main tracking pipeline
│   │   └── team_classifier.py   # Jersey color classification
│   └── utils/
│       ├── __init__.py
│       ├── video_utils.py       # Video processing utilities
│       ├── visualization.py     # Drawing and display functions
│       └── metrics.py           # Performance metrics
├── models/
│   ├── yolo/                    # YOLO model weights (downloaded)
│   ├── sam2/                    # SAM2 model weights (downloaded)
│   └── deepsort/                # DeepSORT feature extractor weights
├── data/
│   ├── sample_videos/           # Sample basketball videos
│   ├── test_images/             # Test images for debugging
│   └── annotations/             # Ground truth annotations (optional)
├── scripts/
│   ├── download_models.py       # Script to download model weights
│   ├── process_video.py         # Main video processing script
│   ├── evaluate_performance.py # Performance evaluation script
│   └── demo.py                  # Quick demo script
├── notebooks/
│   ├── 01_yolo_detection.ipynb  # YOLO detection tutorial
│   ├── 02_sam2_integration.ipynb # SAM2 integration tutorial
│   ├── 03_deepsort_tracking.ipynb # DeepSORT tracking tutorial
│   └── 04_full_pipeline.ipynb   # Complete pipeline tutorial
├── tests/
│   ├── __init__.py
│   ├── test_detection.py        # Unit tests for detection
│   ├── test_segmentation.py     # Unit tests for segmentation
│   ├── test_tracking.py         # Unit tests for tracking
│   └── test_pipeline.py         # Integration tests
├── docs/
│   ├── API.md                   # API documentation
│   ├── PERFORMANCE.md           # Performance benchmarks
│   └── TROUBLESHOOTING.md       # Common issues and solutions
└── docker/
    ├── Dockerfile               # Docker container setup
    ├── docker-compose.yml       # Multi-service setup
    └── requirements-docker.txt  # Docker-specific requirements
```

## 🔧 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/basketball-player-tracking.git
cd basketball-player-tracking
```

### 2. Install Dependencies (using Poetry)
```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install

# Activate the virtual environment
poetry shell
```

### 3. Download Models
```bash
poetry run python scripts/download_models.py
# or if you're in the poetry shell:
python scripts/download_models.py
```

### 4. Run Demo
```bash
poetry run python scripts/demo.py --video data/sample_videos/sample_game.mp4
# or if you're in the poetry shell:
python scripts/demo.py --video data/sample_videos/sample_game.mp4
```

## 🎯 Implementation Phases

This project is designed to be built incrementally:

### Phase 1: YOLO Detection 🎯
- Set up YOLOv11 Nano for player detection
- Implement basic bounding box detection
- Test on basketball videos with elevated camera angles
- **Goal**: 95%+ player detection accuracy

### Phase 2: SAM2 Integration 🖼️
- Add SAM2 for pixel-accurate segmentation
- Integrate with YOLO bounding boxes as prompts
- Optimize for basketball player shapes
- **Goal**: Precise player boundaries for analytics

### Phase 3: DeepSORT Tracking 👥
- Implement DeepSORT for ID consistency
- Add appearance feature extraction
- Handle basketball-specific occlusions
- **Goal**: 85%+ ID retention across frames

### Phase 4: Team Classification 🔴🟢
- Add jersey color detection (Red vs Green)
- Implement temporal smoothing for consistency
- Optimize for basketball lighting conditions
- **Goal**: 90%+ team classification accuracy

### Phase 5: Optimization & Deployment ⚡
- Performance optimization for real-time processing
- Cloud deployment configuration
- Multi-camera support
- **Goal**: 30+ FPS processing speed

## 📋 Requirements

### Hardware
- **GPU**: NVIDIA GTX 1060 or better (RTX 3070+ recommended)
- **RAM**: 8GB minimum (16GB+ recommended)
- **Storage**: 5GB for models and dependencies

### Software
- **Python**: 3.11 (managed by Poetry)
- **Poetry**: For dependency management
- **CUDA**: 11.7+ (for GPU acceleration)
- **FFmpeg**: For video processing

### Key Dependencies
- `torch >= 2.0.0`
- `torchvision >= 0.15.0`
- `ultralytics >= 8.0.0` (YOLO11)
- `opencv-python >= 4.8.0`
- `numpy >= 1.24.0`
- `pillow >= 9.5.0`

## 🎮 Usage Examples

### Basic Video Processing
```python
from src.pipeline.basketball_tracker import BasketballTracker

# Initialize tracker
tracker = BasketballTracker()

# Process video
results = tracker.process_video('input_video.mp4', 'output_video.mp4')

# Get tracking data
player_data = results['tracking_data']
team_assignments = results['team_data']
```

### Real-time Processing
```python
import cv2
from src.pipeline.basketball_tracker import BasketballTracker

tracker = BasketballTracker()
cap = cv2.VideoCapture(0)  # Webcam or video file

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame
    tracked_frame, data = tracker.process_frame(frame)
    
    # Display results
    cv2.imshow('Basketball Tracking', tracked_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

## 📊 Performance Targets

| Metric | Target | Current |
|--------|---------|---------|
| Detection Accuracy | 95%+ | TBD |
| Segmentation Quality | IoU > 0.85 | TBD |
| ID Retention | 85%+ | TBD |
| Team Classification | 90%+ | TBD |
| Processing Speed | 30+ FPS | TBD |
| Multi-player Support | 10+ players | TBD |

## 🤝 Contributing

We welcome contributions! Please see our implementation guide for step-by-step development instructions.

### Development Workflow
1. Follow the [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)
2. Create feature branches for each phase
3. Test thoroughly before submitting PRs
4. Update documentation and benchmarks

### Code Style
- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings to all public functions
- Include unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics**: YOLOv11 implementation
- **Meta**: SAM2 segmentation model  
- **DeepSORT**: Multi-object tracking algorithm
- **Basketball community**: For inspiration and feedback

## 🔗 Related Projects

- [YOLOv11 Official Repository](https://github.com/ultralytics/ultralytics)
- [SAM2 Official Repository](https://github.com/facebookresearch/segment-anything-2)
- [DeepSORT Implementation](https://github.com/nwojke/deep_sort)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/basketball-player-tracking/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/basketball-player-tracking/discussions)
- **Email**: your.email@domain.com

---

**Ready to track some basketball players?** 🏀 Start with the [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)!