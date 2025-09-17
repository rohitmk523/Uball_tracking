# Basketball Player Tracking System

A comprehensive computer vision pipeline for tracking basketball players using **YOLO11 + SAM2 + ByteTrack**, designed for elevated camera angles and multi-player scenarios.

## 🏀 Overview

This project implements a complete basketball player tracking system with three integrated components:

1. **Detection**: YOLO11 (Nano/Small/Medium/Large) for robust player detection
2. **Segmentation**: SAM2 (Tiny/Small/Base/Large) for pixel-perfect player boundaries  
3. **Tracking**: Enhanced ByteTrack with mask features for stable ID retention

Perfect for basketball analytics, game highlights, player performance analysis, and sports research.

## 🚀 Features

- **🎯 Robust Detection**: Multiple YOLO11 models (nano to extra-large) for different speed/accuracy needs
- **🖼️ Pixel-Perfect Segmentation**: SAM2 integration for precise player boundaries
- **🔄 Enhanced Tracking**: ByteTrack with mask features for superior ID retention
- **⚡ Flexible Processing**: Real-time or offline processing with configurable optimization
- **🎬 Video Processing**: Process entire videos or short segments for testing
- **📊 Multiple Methods**: Switch between DeepSORT and ByteTrack tracking algorithms
- **🎛️ Optimization Levels**: Fast/Balanced/Quality presets for different use cases
- **🏀 Basketball Optimized**: Handles occlusions, fast movements, and multi-player scenarios

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

### 2. Set Up Environment
```bash
# Create conda environment with Python 3.11
conda create -n basketball_tracking python=3.11
conda activate basketball_tracking

# Install dependencies
pip install -r requirements.txt

# Install additional libraries for enhanced features
pip install supervision mediapipe
```

### 3. Download Models
```bash
# Download YOLO models
python scripts/download_models.py

# Download SAM2 models (choose size based on your needs)
python scripts/download_sam2_models.py
# Select: 1 (tiny - fastest), 2 (small - balanced), or 4 (large - best quality)
```

### 4. Quick Test
```bash
# Test Phase 1 (YOLO detection only)
python scripts/test_yolo.py

# Test Phase 2 (YOLO + SAM2)
python scripts/test_yolo_sam2.py

# Test Phase 3 (Full pipeline)
python scripts/test_full_pipeline.py
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

## 🎬 Video Processing Guide

### 🚀 Quick Start - Process Short Segments

Perfect for testing and experimentation:

```bash
# Process 10-second segment starting at 30s with ByteTrack (recommended)
python scripts/process_short_segment.py \
    --input your_video.mp4 \
    --start 30 \
    --duration 10 \
    --optimization balanced \
    --method bytetrack

# Fast processing (YOLO only, no SAM2)
python scripts/process_short_segment.py \
    --input your_video.mp4 \
    --start 25 \
    --duration 5 \
    --optimization fast \
    --method bytetrack

# High quality processing (larger models)
python scripts/process_short_segment.py \
    --input your_video.mp4 \
    --start 15 \
    --duration 15 \
    --optimization quality \
    --method bytetrack
```

### 🎯 Full Video Processing

Process entire videos with offline rendering:

```bash
# Process entire video with ByteTrack (recommended)
python scripts/process_and_playback.py \
    --input your_full_video.mp4 \
    --optimization balanced \
    --method bytetrack

# Process only (no playback)
python scripts/process_and_playback.py \
    --input your_full_video.mp4 \
    --optimization balanced \
    --method bytetrack \
    --no-playback

# Playback existing processed video
python scripts/process_and_playback.py \
    --playback-only your_video_tracked_balanced_bytetrack_30fps.mp4
```

### ⚙️ Optimization Levels

Choose the right balance of speed vs quality:

| Level | Models | Speed | Quality | Use Case |
|-------|--------|--------|---------|-----------|
| **fast** | YOLO11s, No SAM2 | ~15-20 FPS | Good | Quick testing, real-time |
| **balanced** | YOLO11m, SAM2-tiny | ~5-10 FPS | Great | Best balance (recommended) |
| **quality** | YOLO11l, SAM2-small | ~2-5 FPS | Excellent | Final production, analysis |

### 🔧 Method Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|-----------|
| **bytetrack** | ✅ Stable IDs<br>✅ Fast<br>✅ Less flickering | ❌ Motion-based only | Basketball (recommended) |
| **deepsort** | ✅ Appearance features<br>✅ Re-identification | ❌ Can flicker<br>❌ Slower | General tracking |

### 📊 Processing Times (Estimated)

For a 60-second basketball video:

| Configuration | Processing Time | Output Quality |
|---------------|----------------|----------------|
| Fast + ByteTrack | ~3-5 minutes | Good for testing |
| Balanced + ByteTrack | ~15-25 minutes | **Recommended** |
| Quality + ByteTrack | ~45-90 minutes | Best for analysis |

### 🎮 Python API Usage

```python
from src.pipeline.basketball_tracker_bytetrack import BasketballTrackerByteTrack

# Initialize tracker
tracker = BasketballTrackerByteTrack(optimization_level="balanced")

# Process single frame
tracked_frame, players = tracker.process_frame(frame)

# Process video file
tracker.process_video("input.mp4", "output.mp4")
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