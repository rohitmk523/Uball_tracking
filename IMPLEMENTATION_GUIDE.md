# Basketball Player Tracking - Implementation Guide

**🎉 IMPLEMENTATION COMPLETE!** This project has been fully implemented with YOLO11 + SAM2 + ByteTrack integration.

## 📋 Current Status

✅ **Phase 1**: YOLO detection with multiple model sizes  
✅ **Phase 2**: SAM2 segmentation integration with mask features  
✅ **Phase 3**: Enhanced ByteTrack tracking with stable ID retention  
✅ **Pipeline**: Complete video processing with optimization levels  
✅ **Scripts**: Full and segment processing with method switching

## 🚀 Quick Start (Already Implemented)

The system is ready to use! See the [VIDEO_PROCESSING_GUIDE.md](VIDEO_PROCESSING_GUIDE.md) for complete usage instructions.

### Test the Implementation

```bash
# Test short segment (recommended first step)
python scripts/process_short_segment.py \
    --input your_video.mp4 \
    --start 30 \
    --duration 10 \
    --optimization balanced \
    --method bytetrack

# Process full video
python scripts/process_and_playback.py \
    --input your_video.mp4 \
    --optimization balanced \
    --method bytetrack
```

---

## 📚 Original Implementation Guide

This guide walks you through building the basketball player tracking system step-by-step, from basic YOLO detection to a complete tracking pipeline.

## 🎯 Phase 1: YOLO Detection Setup

### Step 1.1: Environment Setup

```bash
# Create virtual environment
python -m venv basketball_tracking_env
source basketball_tracking_env/bin/activate  # Linux/Mac
# OR
basketball_tracking_env\Scripts\activate  # Windows

# Install base requirements
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python pillow numpy matplotlib
```

### Step 1.2: Create YOLO Detector Class

**File**: `src/detection/yolo_detector.py`

```python
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

class YOLODetector:
    """YOLOv11 Nano detector optimized for basketball players"""
    
    def __init__(self, model_path: str = "yolo11n.pt", conf_threshold: float = 0.5):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def detect_players(self, image: np.ndarray) -> List[Dict]:
        """
        Detect basketball players in image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of detections with bounding boxes and confidence scores
        """
        # Run YOLO inference
        results = self.model(image, conf=self.conf_threshold, classes=[0])  # class 0 = person
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract box coordinates and confidence
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                        'area': int((x2 - x1) * (y2 - y1))
                    }
                    detections.append(detection)
        
        return detections
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes on image
        
        Args:
            image: Input image
            detections: List of detections from detect_players()
            
        Returns:
            Image with drawn bounding boxes
        """
        vis_image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence score
            label = f"Player: {conf:.2f}"
            cv2.putText(vis_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return vis_image
```

### Step 1.3: Test YOLO Detection

**File**: `scripts/test_yolo.py`

```python
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.yolo_detector import YOLODetector

def test_yolo_on_video(video_path: str):
    """Test YOLO detection on video"""
    detector = YOLODetector()
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect players
        detections = detector.detect_players(frame)
        
        # Visualize
        vis_frame = detector.visualize_detections(frame, detections)
        
        # Display
        cv2.imshow('YOLO Basketball Detection', vis_frame)
        
        # Print detection info
        print(f"Frame: {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}, "
              f"Players detected: {len(detections)}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nFinal team statistics over {frame_count} frames:")
    total = sum(team_stats.values())
    if total > 0:
        for team, count in team_stats.items():
            percentage = (count / total) * 100
            print(f"  {team.upper()}: {count} detections ({percentage:.1f}%)")

if __name__ == "__main__":
    test_team_classification(0)  # Use webcam or video path
```

### Step 4.4: Phase 4 Validation

**Checklist for Phase 4 completion:**
- [ ] Red team players consistently identified with red bounding boxes
- [ ] Green team players consistently identified with green bounding boxes
- [ ] Team assignments remain stable across frames (temporal smoothing works)
- [ ] Classification accuracy >85% for clearly visible jerseys
- [ ] Unknown classification for ambiguous cases works properly

---

## ⚡ Phase 5: Optimization & Final Integration

### Step 5.1: Performance Optimization

**File**: `src/utils/performance_optimizer.py`

```python
import time
import torch
import cv2
import numpy as np
from typing import Dict, List, Optional
from contextlib import contextmanager

class PerformanceOptimizer:
    """Performance optimization utilities for basketball tracking"""
    
    def __init__(self):
        self.timing_stats = {}
        self.frame_skip = 1
        self.resize_factor = 1.0
        self.batch_processing = False
    
    @contextmanager
    def timer(self, operation_name: str):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            if operation_name not in self.timing_stats:
                self.timing_stats[operation_name] = []
            
            self.timing_stats[operation_name].append(duration)
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        stats = {}
        for operation, times in self.timing_stats.items():
            stats[operation] = {
                'avg_time': np.mean(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'total_calls': len(times)
            }
        return stats
    
    def optimize_frame_size(self, frame: np.ndarray, target_width: int = 1280) -> np.ndarray:
        """Optimize frame size for processing"""
        height, width = frame.shape[:2]
        
        if width > target_width:
            self.resize_factor = target_width / width
            new_height = int(height * self.resize_factor)
            return cv2.resize(frame, (target_width, new_height))
        
        return frame
    
    def should_process_frame(self, frame_count: int) -> bool:
        """Determine if frame should be processed (frame skipping)"""
        return frame_count % self.frame_skip == 0
    
    def print_performance_report(self):
        """Print detailed performance report"""
        print("\n" + "="*50)
        print("PERFORMANCE REPORT")
        print("="*50)
        
        stats = self.get_performance_stats()
        
        total_time = 0
        for operation, data in stats.items():
            avg_time = data['avg_time'] * 1000  # Convert to ms
            total_calls = data['total_calls']
            total_op_time = avg_time * total_calls / 1000  # Convert back to seconds
            
            print(f"{operation}:")
            print(f"  Average time: {avg_time:.2f} ms")
            print(f"  Total calls: {total_calls}")
            print(f"  Total time: {total_op_time:.2f} s")
            print()
            
            total_time += total_op_time
        
        print(f"Total processing time: {total_time:.2f} s")
        
        if 'full_pipeline' in stats:
            avg_fps = 1.0 / stats['full_pipeline']['avg_time']
            print(f"Average FPS: {avg_fps:.2f}")

class OptimizedBasketballTracker:
    """Optimized version of BasketballTracker with performance monitoring"""
    
    def __init__(self, optimization_level: str = "balanced"):
        """
        Initialize optimized tracker
        
        Args:
            optimization_level: 'fast', 'balanced', or 'quality'
        """
        from src.detection.yolo_detector import YOLODetector
        from src.segmentation.sam2_segmenter import SAM2Segmenter
        from src.tracking.deepsort_tracker import DeepSORTTracker
        from src.pipeline.team_classifier import TeamClassifier
        
        self.optimizer = PerformanceOptimizer()
        self.optimization_level = optimization_level
        
        # Configure based on optimization level
        if optimization_level == "fast":
            yolo_conf = 0.3
            self.optimizer.frame_skip = 2
            self.target_width = 960
        elif optimization_level == "balanced":
            yolo_conf = 0.5
            self.optimizer.frame_skip = 1
            self.target_width = 1280
        else:  # quality
            yolo_conf = 0.7
            self.optimizer.frame_skip = 1
            self.target_width = 1920
        
        # Initialize components
        with self.optimizer.timer('initialization'):
            self.detector = YOLODetector(conf_threshold=yolo_conf)
            self.segmenter = SAM2Segmenter()
            self.tracker = DeepSORTTracker()
            self.team_classifier = TeamClassifier()
        
        # Performance tracking
        self.frame_count = 0
        self.processed_frames = 0
        
        # Visualization colors
        self.team_colors = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'unknown': (128, 128, 128)
        }
    
    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, List[Dict]]:
        """Process frame with performance optimization"""
        with self.optimizer.timer('full_pipeline'):
            # Check if we should process this frame
            if not self.optimizer.should_process_frame(self.frame_count):
                self.frame_count += 1
                return frame, []  # Return original frame with no processing
            
            # Optimize frame size
            with self.optimizer.timer('frame_preprocessing'):
                optimized_frame = self.optimizer.optimize_frame_size(frame, self.target_width)
            
            # Step 1: Detection
            with self.optimizer.timer('yolo_detection'):
                detections = self.detector.detect_players(optimized_frame)
            
            # Step 2: Segmentation (skip if no detections)
            segmented_detections = []
            if detections:
                with self.optimizer.timer('sam2_segmentation'):
                    segmented_detections = self.segmenter.segment_players(optimized_frame, detections)
            
            # Step 3: Tracking
            with self.optimizer.timer('deepsort_tracking'):
                tracked_objects = self.tracker.update(segmented_detections, optimized_frame)
            
            # Step 4: Team classification
            with self.optimizer.timer('team_classification'):
                classified_objects = self.team_classifier.classify_teams(
                    optimized_frame, tracked_objects, segmented_detections)
            
            # Step 5: Visualization
            with self.optimizer.timer('visualization'):
                vis_frame = self.visualize_results(optimized_frame, classified_objects, segmented_detections)
                
                # Resize back to original size if needed
                if self.optimizer.resize_factor < 1.0:
                    original_height, original_width = frame.shape[:2]
                    vis_frame = cv2.resize(vis_frame, (original_width, original_height))
            
            self.frame_count += 1
            self.processed_frames += 1
            
            return vis_frame, classified_objects
    
    def visualize_results(self, frame: np.ndarray, tracked_objects: List[Dict], 
                         detections: List[Dict]) -> np.ndarray:
        """Optimized visualization"""
        vis_frame = frame.copy()
        
        # Create bbox to mask mapping
        bbox_to_mask = {tuple(det['bbox']): det.get('mask') 
                       for det in detections if 'mask' in det}
        
        for obj in tracked_objects:
            track_id = obj['track_id']
            bbox = obj['bbox']
            team = obj.get('team', 'unknown')
            x1, y1, x2, y2 = bbox
            
            color = self.team_colors[team]
            
            # Draw mask if available (lighter weight)
            bbox_key = tuple(bbox)
            if bbox_key in bbox_to_mask and bbox_to_mask[bbox_key] is not None:
                mask = bbox_to_mask[bbox_key]
                mask_indices = np.where(mask)
                if len(mask_indices[0]) > 0:
                    vis_frame[mask_indices] = (
                        vis_frame[mask_indices] * 0.7 + 
                        np.array(color) * 0.3
                    ).astype(np.uint8)
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw simplified label
            label = f"#{track_id} {team[0].upper()}"
            cv2.putText(vis_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add performance info
        if self.frame_count % 30 == 0 and self.optimizer.timing_stats:
            avg_time = np.mean(self.optimizer.timing_stats.get('full_pipeline', [0.1]))
            fps = 1.0 / avg_time
            perf_text = f"FPS: {fps:.1f} | Level: {self.optimization_level}"
            cv2.putText(vis_frame, perf_text, (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_frame
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        stats = self.optimizer.get_performance_stats()
        
        summary = {
            'optimization_level': self.optimization_level,
            'total_frames': self.frame_count,
            'processed_frames': self.processed_frames,
            'frame_skip_ratio': 1 - (self.processed_frames / max(self.frame_count, 1)),
            'resize_factor': self.optimizer.resize_factor,
            'component_timings': {}
        }
        
        for component, data in stats.items():
            summary['component_timings'][component] = {
                'avg_ms': data['avg_time'] * 1000,
                'calls': data['total_calls']
            }
        
        if 'full_pipeline' in stats:
            summary['avg_fps'] = 1.0 / stats['full_pipeline']['avg_time']
        
        return summary
```

### Step 5.2: Create Main Application Script

**File**: `scripts/basketball_tracker_app.py`

```python
import cv2
import argparse
import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.performance_optimizer import OptimizedBasketballTracker

def main():
    parser = argparse.ArgumentParser(description='Basketball Player Tracking Application')
    parser.add_argument('--input', '-i', type=str, default='0', 
                       help='Input source (webcam=0, video file path, or RTSP stream)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output video file path (optional)')
    parser.add_argument('--optimization', '-opt', choices=['fast', 'balanced', 'quality'], 
                       default='balanced', help='Optimization level')
    parser.add_argument('--stats', '-s', action='store_true',
                       help='Save performance statistics')
    parser.add_argument('--no-display', action='store_true',
                       help='Run without GUI display (for headless servers)')
    
    args = parser.parse_args()
    
    # Initialize tracker
    print(f"Initializing Basketball Tracker (optimization: {args.optimization})...")
    tracker = OptimizedBasketballTracker(optimization_level=args.optimization)
    
    # Setup input
    if args.input.isdigit():
        input_source = int(args.input)
        print(f"Using webcam {input_source}")
    else:
        input_source = args.input
        print(f"Using video file: {input_source}")
    
    cap = cv2.VideoCapture(input_source)
    
    if not cap.isOpened():
        print(f"Error: Could not open input source {input_source}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {width}x{height} @ {fps} FPS")
    
    # Setup output writer if specified
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Recording to: {args.output}")
    
    # Main processing loop
    frame_count = 0
    print("\nStarting processing... Press 'q' to quit, 's' to save stats")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or camera disconnected")
                break
            
            # Process frame
            vis_frame, tracked_objects = tracker.process_frame(frame)
            
            # Write to output if specified
            if out:
                out.write(vis_frame)
            
            # Display (if not headless)
            if not args.no_display:
                cv2.imshow('Basketball Player Tracking', vis_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quit requested by user")
                    break
                elif key == ord('s'):
                    save_performance_stats(tracker, f"stats_frame_{frame_count}.json")
            
            # Periodic progress updates
            if frame_count % 100 == 0 and frame_count > 0:
                print(f"Processed {frame_count} frames...")
                if tracked_objects:
                    teams = {}
                    for obj in tracked_objects:
                        team = obj.get('team', 'unknown')
                        teams[team] = teams.get(team, 0) + 1
                    print(f"  Current players: {teams}")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if out:
            out.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        
        # Final statistics
        print(f"\nProcessing complete: {frame_count} frames processed")
        tracker.optimizer.print_performance_report()
        
        # Save statistics if requested
        if args.stats:
            save_performance_stats(tracker, "final_performance_stats.json")

def save_performance_stats(tracker, filename: str):
    """Save performance statistics to JSON file"""
    stats = tracker.get_performance_summary()
    
    with open(filename, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Performance statistics saved to: {filename}")

if __name__ == "__main__":
    main()
```

### Step 5.3: Create Installation and Setup Scripts

**File**: `scripts/setup.py`

```python
#!/usr/bin/env python3
import os
import sys
import subprocess
import urllib.request
from pathlib import Path

def run_command(command, description):
    """Run a shell command with error handling"""
    print(f"\n{description}...")
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"✅ {description} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False
    return True

def download_file(url, filepath):
    """Download a file from URL"""
    print(f"Downloading {filepath.name}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"✅ Downloaded {filepath.name}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filepath.name}: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        'models/yolo',
        'models/sam2',
        'models/deepsort',
        'data/sample_videos',
        'data/test_images',
        'outputs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def setup_python_environment():
    """Setup Python environment and install dependencies"""
    print("Setting up Python environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    print(f"✅ Python version: {sys.version}")
    
    # Install requirements
    requirements = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pillow>=9.5.0",
        "matplotlib>=3.7.0",
        "filterpy",
        "scipy",
        "lap",
        "easydict"
    ]
    
    for requirement in requirements:
        if not run_command(f"pip install {requirement}", f"Installing {requirement}"):
            return False
    
    return True

def download_models():
    """Download pre-trained models"""
    print("\nDownloading pre-trained models...")
    
    models_dir = Path("models")
    
    # YOLO models - these will be downloaded automatically by ultralytics
    print("📦 YOLO models will be downloaded automatically on first use")
    
    # SAM2 models
    print("📦 SAM2 models will be downloaded automatically on first use")
    
    # Create placeholder files to indicate model directories
    (models_dir / "yolo" / "README.md").write_text("YOLO models will be downloaded here automatically")
    (models_dir / "sam2" / "README.md").write_text("SAM2 models will be downloaded here automatically")
    (models_dir / "deepsort" / "README.md").write_text("DeepSORT models will be downloaded here automatically")
    
    return True

def create_sample_scripts():
    """Create sample test scripts"""
    sample_test = '''#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.basketball_tracker_app import main

if __name__ == "__main__":
    print("Basketball Player Tracking - Quick Test")
    print("This will use your webcam. Press 'q' to quit.")
    
    # Override sys.argv for webcam test
    sys.argv = ['test_quick.py', '--input', '0', '--optimization', 'fast']
    main()
'''
    
    Path("test_quick.py").write_text(sample_test)
    print("📝 Created test_quick.py")

def main():
    """Main setup function"""
    print("🏀 Basketball Player Tracking System Setup")
    print("=" * 50)
    
    # Create directories
    print("\n1. Creating project directories...")
    create_directories()
    
    # Setup Python environment
    print("\n2. Setting up Python environment...")
    if not setup_python_environment():
        print("❌ Failed to setup Python environment")
        return False
    
    # Download models
    print("\n3. Setting up models...")
    if not download_models():
        print("❌ Failed to download models")
        return False
    
    # Create sample scripts
    print("\n4. Creating sample scripts...")
    create_sample_scripts()
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Run 'python test_quick.py' to test with webcam")
    print("2. Run 'python scripts/basketball_tracker_app.py --help' for all options")
    print("3. Check the implementation guide for detailed usage instructions")
    
    return True

if __name__ == "__main__":
    main()
```

### Step 5.4: Create requirements.txt

**File**: `requirements.txt`

```
# Core ML libraries
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0

# Computer vision
opencv-python>=4.8.0
pillow>=9.5.0

# YOLO
ultralytics>=8.0.0

# SAM2 (will be installed via git)
# git+https://github.com/facebookresearch/segment-anything-2.git

# Tracking
filterpy
scipy
lap
easydict

# Visualization
matplotlib>=3.7.0

# Utilities
pathlib
argparse
json5
tqdm

# Optional: for better performance
# onnxruntime-gpu  # For ONNX model inference
# tensorrt  # For TensorRT optimization (NVIDIA only)
```

### Step 5.5: Final Testing and Documentation

**File**: `scripts/run_benchmarks.py`

```python
#!/usr/bin/env python3
import cv2
import time
import sys
import os
import numpy as np
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.performance_optimizer import OptimizedBasketballTracker

def run_benchmark_test(optimization_level: str = "balanced", duration: int = 60):
    """Run benchmark test for specified duration"""
    print(f"\n🏀 Running benchmark test: {optimization_level} optimization")
    print(f"Duration: {duration} seconds")
    print("-" * 50)
    
    # Initialize tracker
    tracker = OptimizedBasketballTracker(optimization_level=optimization_level)
    
    # Use webcam or test video
    cap = cv2.VideoCapture(0)  # Change to video file if needed
    
    if not cap.isOpened():
        print("❌ Could not open camera/video")
        return None
    
    start_time = time.time()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        vis_frame, tracked_objects = tracker.process_frame(frame)
        
        # Display (optional - comment out for headless testing)
        cv2.imshow(f'Benchmark - {optimization_level}', vis_frame)
        
        frame_count += 1
        elapsed = time.time() - start_time
        
        # Print periodic updates
        if frame_count % 30 == 0:
            fps = frame_count / elapsed
            print(f"Frame {frame_count}: {fps:.1f} FPS, Players: {len(tracked_objects)}")
        
        # Break after specified duration
        if elapsed >= duration:
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Calculate final statistics
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time
    
    print(f"\n📊 Benchmark Results - {optimization_level}:")
    print(f"Total frames: {frame_count}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")
    
    # Get detailed performance stats
    performance_summary = tracker.get_performance_summary()
    
    return {
        'optimization_level': optimization_level,
        'total_frames': frame_count,
        'total_time': total_time,
        'avg_fps': avg_fps,
        'performance_details': performance_summary
    }

def main():
    """Run comprehensive benchmark tests"""
    print("🏀 Basketball Player Tracking - Performance Benchmarks")
    print("=" * 60)
    
    optimization_levels = ['fast', 'balanced', 'quality']
    results = {}
    
    for level in optimization_levels:
        result = run_benchmark_test(level, duration=30)  # 30 seconds per test
        if result:
            results[level] = result
    
    # Print comparison
    print("\n📈 Benchmark Comparison:")
    print("-" * 60)
    print(f"{'Level':<10} {'FPS':<8} {'Frames':<8} {'Time':<8}")
    print("-" * 60)
    
    for level, result in results.items():
        print(f"{level:<10} {result['avg_fps']:<8.1f} {result['total_frames']:<8} {result['total_time']:<8.1f}")
    
    print("\n✅ Benchmark tests completed!")
    print("\nRecommendations:")
    if results:
        best_fps = max(results.values(), key=lambda x: x['avg_fps'])
        print(f"🏃‍♂️ Best FPS: {best_fps['optimization_level']} ({best_fps['avg_fps']:.1f} FPS)")
        
        if best_fps['avg_fps'] >= 30:
            print("✅ Real-time performance achieved!")
        elif best_fps['avg_fps'] >= 15:
            print("⚠️  Near real-time performance - consider GPU upgrade")
        else:
            print("❌ Performance below real-time - optimization needed")

if __name__ == "__main__":
    main()
```

### Step 5.6: Phase 5 Validation

**Checklist for Phase 5 completion:**
- [ ] System achieves target FPS on available hardware (>15 FPS minimum)
- [ ] All components work together smoothly in the complete pipeline
- [ ] Performance optimization provides measurable improvements
- [ ] Setup script successfully installs all dependencies
- [ ] Benchmark tests run successfully and provide useful metrics
- [ ] Code is well-documented and organized
- [ ] Repository structure matches the planned layout

---

## 🎯 Implementation Summary

This implementation guide provides a complete, step-by-step approach to building a basketball player tracking system:

### **Phase 1**: YOLO Detection Foundation
- Set up YOLOv11 Nano for fast, accurate player detection
- Handle basketball-specific scenarios (elevated cameras, multiple players)
- Achieve >90% detection accuracy

### **Phase 2**: SAM2 Segmentation Integration  
- Add pixel-accurate player segmentation using SAM2
- Use YOLO bounding boxes as prompts for SAM2
- Maintain reasonable processing speed (>10 FPS)

### **Phase 3**: DeepSORT Tracking Implementation
- Implement robust player ID tracking across frames
- Handle occlusions and player interactions
- Achieve >80% ID retention rate

### **Phase 4**: Team Classification
- Add red vs green team identification
- Implement temporal smoothing for stable assignments
- Achieve >85% team classification accuracy

### **Phase 5**: Optimization & Production
- Performance optimization for real-time processing
- Complete application with GUI and CLI interfaces
- Comprehensive testing and benchmarking tools

### Key Success Metrics:
- **Detection**: >95% player detection accuracy
- **Segmentation**: IoU >0.85 for player masks  
- **Tracking**: >85% cross-frame ID retention
- **Team Classification**: >90% accuracy with temporal smoothing
- **Performance**: >15 FPS processing speed (target: 30+ FPS)
- **Multi-player**: Handle 10+ simultaneous players

### Next Steps After Implementation:
1. **Dataset Collection**: Gather more basketball-specific training data
2. **Cross-camera Integration**: Extend to multiple camera views
3. **Advanced Analytics**: Add player statistics and game analysis
4. **Cloud Deployment**: Scale to production cloud infrastructure
5. **Mobile Integration**: Develop mobile app interfaces

This guide provides everything needed to build a production-ready basketball player tracking system from scratch! 🏀xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Test with webcam or video file
    test_yolo_on_video(0)  # Use 0 for webcam, or path to video file
```

### Step 1.4: Phase 1 Validation

**Checklist for Phase 1 completion:**
- [ ] YOLO detects players with >90% accuracy on basketball videos
- [ ] Handles elevated camera angles properly
- [ ] Processes at least 15+ FPS on available hardware
- [ ] False positive rate <10% (not detecting non-players as players)
- [ ] Bounding boxes properly encompass player bodies

---

## 🖼️ Phase 2: SAM2 Integration

### Step 2.1: Install SAM2 Dependencies

```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
pip install matplotlib pillow
```

### Step 2.2: Create SAM2 Segmenter Class

**File**: `src/segmentation/sam2_segmenter.py`

```python
import torch
import numpy as np
import cv2
from typing import List, Dict, Optional
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class SAM2Segmenter:
    """SAM2 segmentation for basketball players"""
    
    def __init__(self, model_cfg: str = "sam2_hiera_l.yaml", checkpoint: str = "sam2_hiera_large.pt"):
        """
        Initialize SAM2 segmenter
        
        Args:
            model_cfg: SAM2 model configuration
            checkpoint: Path to SAM2 checkpoint
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build SAM2 model
        sam2_model = build_sam2(model_cfg, checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        
        print(f"SAM2 initialized on device: {self.device}")
    
    def segment_players(self, image: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        Create segmentation masks for detected players
        
        Args:
            image: Input image (BGR format)
            detections: List of player detections from YOLO
            
        Returns:
            List of detections with added segmentation masks
        """
        # Set image for SAM2
        self.predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        segmented_detections = []
        
        for detection in detections:
            # Use bounding box as prompt for SAM2
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Convert to SAM2 box format (x, y, w, h)
            input_box = np.array([x1, y1, x2, y2])
            
            # Get segmentation mask
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            
            # Add mask to detection
            segmented_detection = detection.copy()
            segmented_detection['mask'] = masks[0]  # Take the best mask
            segmented_detection['mask_score'] = float(scores[0])
            
            segmented_detections.append(segmented_detection)
        
        return segmented_detections
    
    def visualize_segmentation(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Visualize segmentation masks on image
        
        Args:
            image: Input image
            detections: List of detections with masks
            
        Returns:
            Image with segmentation overlays
        """
        vis_image = image.copy()
        
        for i, detection in enumerate(detections):
            if 'mask' not in detection:
                continue
                
            mask = detection['mask']
            bbox = detection['bbox']
            conf = detection['confidence']
            
            # Create colored mask overlay
            color = np.array([0, 255, 0])  # Green
            colored_mask = np.zeros_like(vis_image)
            colored_mask[mask] = color
            
            # Blend mask with image
            vis_image = cv2.addWeighted(vis_image, 0.8, colored_mask, 0.4, 0)
            
            # Draw bounding box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"Player {i+1}: {conf:.2f}"
            cv2.putText(vis_image, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return vis_image
```

### Step 2.3: Integrate YOLO + SAM2

**File**: `src/pipeline/yolo_sam2_pipeline.py`

```python
import cv2
import numpy as np
from typing import List, Dict
from src.detection.yolo_detector import YOLODetector
from src.segmentation.sam2_segmenter import SAM2Segmenter

class YOLOSAMPipeline:
    """Combined YOLO detection + SAM2 segmentation pipeline"""
    
    def __init__(self):
        self.detector = YOLODetector()
        self.segmenter = SAM2Segmenter()
    
    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, List[Dict]]:
        """
        Process frame with detection and segmentation
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (visualized_frame, detection_data)
        """
        # Step 1: Detect players with YOLO
        detections = self.detector.detect_players(frame)
        
        # Step 2: Segment players with SAM2
        if detections:
            segmented_detections = self.segmenter.segment_players(frame, detections)
        else:
            segmented_detections = []
        
        # Step 3: Visualize results
        vis_frame = self.segmenter.visualize_segmentation(frame, segmented_detections)
        
        return vis_frame, segmented_detections
```

### Step 2.4: Test YOLO + SAM2 Integration

**File**: `scripts/test_yolo_sam2.py`

```python
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.yolo_sam2_pipeline import YOLOSAMPipeline

def test_pipeline_on_video(video_path: str):
    """Test YOLO + SAM2 pipeline on video"""
    pipeline = YOLOSAMPipeline()
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        vis_frame, detections = pipeline.process_frame(frame)
        
        # Display
        cv2.imshow('YOLO + SAM2 Basketball Tracking', vis_frame)
        
        # Print info
        print(f"Frame: {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}, "
              f"Players segmented: {len(detections)}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_pipeline_on_video(0)  # Use webcam or video path
```

### Step 2.5: Phase 2 Validation

**Checklist for Phase 2 completion:**
- [ ] SAM2 creates accurate player silhouettes from YOLO bounding boxes
- [ ] Segmentation quality is visually good (follows player body outline)
- [ ] Processing speed remains >10 FPS
- [ ] Masks properly exclude background and other players
- [ ] Integration between YOLO and SAM2 works smoothly

---

## 👥 Phase 3: DeepSORT Tracking Integration

### Step 3.1: Install DeepSORT Dependencies

```bash
pip install filterpy lap scipy
pip install easydict
```

### Step 3.2: Create DeepSORT Tracker Class

**File**: `src/tracking/deepsort_tracker.py`

```python
import numpy as np
import cv2
from typing import List, Dict, Optional
from collections import defaultdict
import torch
import torch.nn as nn
from filterpy.kalman import KalmanFilter

class FeatureExtractor(nn.Module):
    """Simple CNN feature extractor for DeepSORT"""
    
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256)
        )
    
    def forward(self, x):
        return self.cnn(x)

class Track:
    """Individual track for a basketball player"""
    
    def __init__(self, track_id: int, bbox: List[int], feature: np.ndarray):
        self.track_id = track_id
        self.bbox = bbox
        self.feature = feature
        self.age = 0
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0
        
        # Initialize Kalman filter for position tracking
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.x[:4] = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        
        # Set up state transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # Process and measurement noise
        self.kf.R *= 10.0
        self.kf.Q[4:, 4:] *= 10.0
    
    def predict(self):
        """Predict next position using Kalman filter"""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
    
    def update(self, bbox: List[int], feature: np.ndarray):
        """Update track with new detection"""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.bbox = bbox
        self.feature = feature
        
        # Update Kalman filter
        measurement = np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
        self.kf.update(measurement)
    
    def get_state(self) -> List[int]:
        """Get current bounding box prediction"""
        state = self.kf.x[:4].copy()
        bbox = [int(state[0]), int(state[1]), int(state[0] + state[2]), int(state[1] + state[3])]
        return bbox

class DeepSORTTracker:
    """DeepSORT tracker for basketball players"""
    
    def __init__(self, max_age: int = 30, min_hits: int = 3):
        """
        Initialize DeepSORT tracker
        
        Args:
            max_age: Maximum age of track before deletion
            min_hits: Minimum hits before track is confirmed
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: List[Track] = []
        self.track_id_counter = 0
        
        # Feature extractor for appearance similarity
        self.feature_extractor = FeatureExtractor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
        print(f"DeepSORT tracker initialized on device: {self.device}")
    
    def extract_features(self, image: np.ndarray, bboxes: List[List[int]]) -> List[np.ndarray]:
        """Extract appearance features from detected players"""
        features = []
        
        with torch.no_grad():
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                
                # Crop player region
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    features.append(np.zeros(256))
                    continue
                
                # Resize and normalize
                crop = cv2.resize(crop, (64, 128))
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop = crop.astype(np.float32) / 255.0
                
                # Convert to tensor
                crop_tensor = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0)
                crop_tensor = crop_tensor.to(self.device)
                
                # Extract features
                feature = self.feature_extractor(crop_tensor)
                feature = feature.cpu().numpy().flatten()
                
                # L2 normalize
                feature = feature / (np.linalg.norm(feature) + 1e-8)
                features.append(feature)
        
        return features
    
    def calculate_distances(self, features1: List[np.ndarray], features2: List[np.ndarray]) -> np.ndarray:
        """Calculate cosine distances between feature sets"""
        if not features1 or not features2:
            return np.full((len(features1), len(features2)), 1.0)
        
        distances = np.zeros((len(features1), len(features2)))
        
        for i, f1 in enumerate(features1):
            for j, f2 in enumerate(features2):
                # Cosine distance (1 - cosine similarity)
                similarity = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8)
                distances[i, j] = 1 - similarity
        
        return distances
    
    def associate_detections_to_tracks(self, detections: List[Dict], image: np.ndarray) -> tuple:
        """Associate detections to existing tracks using Hungarian algorithm"""
        if not self.tracks:
            return [], list(range(len(detections))), []
        
        # Extract bounding boxes and features
        det_bboxes = [det['bbox'] for det in detections]
        det_features = self.extract_features(image, det_bboxes)
        
        # Get track features and predicted positions
        track_features = [track.feature for track in self.tracks]
        track_bboxes = [track.get_state() for track in self.tracks]
        
        # Calculate appearance distance matrix
        appearance_distances = self.calculate_distances(track_features, det_features)
        
        # Calculate IoU distance matrix
        iou_distances = np.zeros((len(self.tracks), len(detections)))
        for t, track_bbox in enumerate(track_bboxes):
            for d, det_bbox in enumerate(det_bboxes):
                iou = self.calculate_iou(track_bbox, det_bbox)
                iou_distances[t, d] = 1 - iou
        
        # Combine distances (weighted)
        distance_matrix = 0.7 * appearance_distances + 0.3 * iou_distances
        
        # Use simple greedy assignment (can be replaced with Hungarian algorithm)
        matches, unmatched_dets, unmatched_trks = self.greedy_assignment(distance_matrix, 0.5)
        
        return matches, unmatched_dets, unmatched_trks
    
    def calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def greedy_assignment(self, distance_matrix: np.ndarray, threshold: float) -> tuple:
        """Simple greedy assignment algorithm"""
        matches = []
        unmatched_dets = list(range(distance_matrix.shape[1]))
        unmatched_trks = list(range(distance_matrix.shape[0]))
        
        while True:
            if len(unmatched_trks) == 0 or len(unmatched_dets) == 0:
                break
            
            # Find minimum distance
            min_val = float('inf')
            min_i, min_j = -1, -1
            
            for i in unmatched_trks:
                for j in unmatched_dets:
                    if distance_matrix[i, j] < min_val:
                        min_val = distance_matrix[i, j]
                        min_i, min_j = i, j
            
            if min_val > threshold:
                break
            
            matches.append([min_i, min_j])
            unmatched_trks.remove(min_i)
            unmatched_dets.remove(min_j)
        
        return matches, unmatched_dets, unmatched_trks
    
    def update(self, detections: List[Dict], image: np.ndarray) -> List[Dict]:
        """Update tracker with new detections"""
        # Predict all existing tracks
        for track in self.tracks:
            track.predict()
        
        # Associate detections to tracks
        matches, unmatched_dets, unmatched_trks = self.associate_detections_to_tracks(detections, image)
        
        # Update matched tracks
        for match in matches:
            track_idx, det_idx = match
            det_bbox = detections[det_idx]['bbox']
            det_features = self.extract_features(image, [det_bbox])
            
            self.tracks[track_idx].update(det_bbox, det_features[0])
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det_bbox = detections[det_idx]['bbox']
            det_features = self.extract_features(image, [det_bbox])
            
            new_track = Track(self.track_id_counter, det_bbox, det_features[0])
            self.tracks.append(new_track)
            self.track_id_counter += 1
        
        # Remove old tracks
        self.tracks = [track for track in self.tracks 
                      if track.time_since_update < self.max_age]
        
        # Prepare output
        tracked_objects = []
        for track in self.tracks:
            if track.hits >= self.min_hits:
                tracked_obj = {
                    'track_id': track.track_id,
                    'bbox': track.get_state(),
                    'confidence': 1.0,  # Track confidence (can be improved)
                    'age': track.age,
                    'hits': track.hits
                }
                tracked_objects.append(tracked_obj)
        
        return tracked_objects
```

### Step 3.3: Create Complete Tracking Pipeline

**File**: `src/pipeline/basketball_tracker.py`

```python
import cv2
import numpy as np
from typing import List, Dict, Tuple
from src.detection.yolo_detector import YOLODetector
from src.segmentation.sam2_segmenter import SAM2Segmenter
from src.tracking.deepsort_tracker import DeepSORTTracker

class BasketballTracker:
    """Complete basketball player tracking pipeline"""
    
    def __init__(self):
        """Initialize all components"""
        self.detector = YOLODetector()
        self.segmenter = SAM2Segmenter()
        self.tracker = DeepSORTTracker()
        
        # Color palette for different player IDs
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (0, 128, 0), (128, 128, 0), (0, 128, 128), (128, 0, 0)
        ]
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process single frame through complete pipeline
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (visualized_frame, tracking_data)
        """
        # Step 1: Detect players with YOLO
        detections = self.detector.detect_players(frame)
        
        # Step 2: Segment players with SAM2
        if detections:
            segmented_detections = self.segmenter.segment_players(frame, detections)
        else:
            segmented_detections = []
        
        # Step 3: Track players with DeepSORT
        tracked_objects = self.tracker.update(segmented_detections, frame)
        
        # Step 4: Visualize results
        vis_frame = self.visualize_tracking(frame, tracked_objects, segmented_detections)
        
        return vis_frame, tracked_objects
    
    def visualize_tracking(self, frame: np.ndarray, tracked_objects: List[Dict], 
                          detections: List[Dict]) -> np.ndarray:
        """
        Visualize tracking results on frame
        
        Args:
            frame: Input frame
            tracked_objects: List of tracked objects from DeepSORT
            detections: Original detections with masks
            
        Returns:
            Visualized frame
        """
        vis_frame = frame.copy()
        
        # Create a mapping from bbox to mask
        bbox_to_mask = {}
        for det in detections:
            if 'mask' in det:
                bbox_key = tuple(det['bbox'])
                bbox_to_mask[bbox_key] = det['mask']
        
        for tracked_obj in tracked_objects:
            track_id = tracked_obj['track_id']
            bbox = tracked_obj['bbox']
            x1, y1, x2, y2 = bbox
            
            # Get color for this track
            color = self.colors[track_id % len(self.colors)]
            
            # Draw segmentation mask if available
            bbox_key = tuple(bbox)
            if bbox_key in bbox_to_mask:
                mask = bbox_to_mask[bbox_key]
                colored_mask = np.zeros_like(vis_frame)
                colored_mask[mask] = color
                vis_frame = cv2.addWeighted(vis_frame, 0.8, colored_mask, 0.3, 0)
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw track ID
            label = f"Player {track_id}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Draw label background
            cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(vis_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw center point
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(vis_frame, (center_x, center_y), 5, color, -1)
        
        # Add frame info
        info_text = f"Players tracked: {len(tracked_objects)}"
        cv2.putText(vis_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_frame
    
    def process_video(self, input_path: str, output_path: str = None) -> Dict:
        """
        Process entire video through tracking pipeline
        
        Args:
            input_path: Path to input video
            output_path: Path to save output video (optional)
            
        Returns:
            Dictionary with tracking statistics
        """
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer if output path provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process video
        frame_count = 0
        tracking_data = []
        
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            vis_frame, tracked_objects = self.process_frame(frame)
            
            # Save tracking data
            frame_data = {
                'frame': frame_count,
                'tracked_objects': tracked_objects,
                'player_count': len(tracked_objects)
            }
            tracking_data.append(frame_data)
            
            # Write output frame
            if out:
                out.write(vis_frame)
            
            # Progress update
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% - Players: {len(tracked_objects)}")
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        
        # Calculate statistics
        stats = {
            'total_frames': frame_count,
            'tracking_data': tracking_data,
            'average_players': np.mean([fd['player_count'] for fd in tracking_data]),
            'max_players': max([fd['player_count'] for fd in tracking_data]),
            'unique_track_ids': len(set([obj['track_id'] 
                                        for fd in tracking_data 
                                        for obj in fd['tracked_objects']]))
        }
        
        print(f"\nProcessing complete!")
        print(f"Average players per frame: {stats['average_players']:.1f}")
        print(f"Maximum players tracked: {stats['max_players']}")
        print(f"Unique player IDs: {stats['unique_track_ids']}")
        
        return stats
```

### Step 3.4: Test Complete Pipeline

**File**: `scripts/test_full_pipeline.py`

```python
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.basketball_tracker import BasketballTracker

def test_real_time_tracking(video_path: str):
    """Test complete tracking pipeline in real-time"""
    tracker = BasketballTracker()
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        vis_frame, tracked_objects = tracker.process_frame(frame)
        
        # Display
        cv2.imshow('Basketball Player Tracking', vis_frame)
        
        # Print tracking info
        if frame_count % 10 == 0:  # Print every 10 frames
            print(f"Frame {frame_count}: {len(tracked_objects)} players tracked")
            for obj in tracked_objects:
                print(f"  - Player {obj['track_id']}: bbox={obj['bbox']}")
        
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def test_video_processing(input_video: str, output_video: str):
    """Test complete video processing"""
    tracker = BasketballTracker()
    
    # Process entire video
    stats = tracker.process_video(input_video, output_video)
    
    print(f"\nVideo processing statistics:")
    for key, value in stats.items():
        if key != 'tracking_data':  # Don't print the full data
            print(f"{key}: {value}")

if __name__ == "__main__":
    # Test real-time tracking
    print("Testing real-time tracking...")
    test_real_time_tracking(0)  # Use webcam or video path
    
    # Test video processing
    # test_video_processing("input.mp4", "output.mp4")
```

### Step 3.5: Phase 3 Validation

**Checklist for Phase 3 completion:**
- [ ] Players maintain consistent IDs across frames (>80% retention)
- [ ] New players get assigned unique IDs automatically
- [ ] Lost players are removed after reasonable timeout
- [ ] Tracking works through occlusions and player interactions
- [ ] Processing speed remains acceptable (>5 FPS for development)
- [ ] Visual output clearly shows player IDs and tracking trails

---

## 🔴🟢 Phase 4: Team Classification (Red vs Green)

### Step 4.1: Create Team Classifier

**File**: `src/pipeline/team_classifier.py`

```python
import cv2
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter

class TeamClassifier:
    """Classify basketball players into teams based on jersey colors"""
    
    def __init__(self):
        """Initialize team classifier"""
        # Define color ranges for team classification (HSV)
        self.red_ranges = [
            (np.array([0, 50, 50]), np.array([10, 255, 255])),    # Red range 1
            (np.array([170, 50, 50]), np.array([180, 255, 255]))  # Red range 2
        ]
        
        self.green_ranges = [
            (np.array([40, 50, 50]), np.array([80, 255, 255]))    # Green range
        ]
        
        # Team history for temporal smoothing
        self.team_history = {}  # track_id -> list of recent team assignments
        self.history_length = 10
    
    def extract_jersey_region(self, image: np.ndarray, bbox: List[int], 
                             mask: np.ndarray = None) -> np.ndarray:
        """
        Extract jersey region from player detection
        
        Args:
            image: Full frame image
            bbox: Player bounding box [x1, y1, x2, y2]
            mask: Player segmentation mask (optional)
            
        Returns:
            Jersey region image
        """
        x1, y1, x2, y2 = bbox
        
        # Crop player region
        player_crop = image[y1:y2, x1:x2]
        
        if mask is not None:
            # Use mask to focus on player pixels
            mask_crop = mask[y1:y2, x1:x2]
            player_crop = cv2.bitwise_and(player_crop, player_crop, mask=mask_crop.astype(np.uint8))
        
        # Focus on upper torso (jersey area)
        height, width = player_crop.shape[:2]
        torso_y1 = int(height * 0.2)  # Skip head area
        torso_y2 = int(height * 0.7)  # Focus on torso
        
        jersey_region = player_crop[torso_y1:torso_y2, :]
        
        return jersey_region
    
    def classify_jersey_color(self, jersey_region: np.ndarray) -> Tuple[str, float]:
        """
        Classify jersey color as red, green, or unknown
        
        Args:
            jersey_region: Cropped jersey region
            
        Returns:
            Tuple of (team_name, confidence)
        """
        if jersey_region.size == 0:
            return "unknown", 0.0
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)
        
        # Calculate color masks
        red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low, high in self.red_ranges:
            mask = cv2.inRange(hsv, low, high)
            red_mask = cv2.bitwise_or(red_mask, mask)
        
        green_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low, high in self.green_ranges:
            mask = cv2.inRange(hsv, low, high)
            green_mask = cv2.bitwise_or(green_mask, mask)
        
        # Calculate color percentages
        total_pixels = hsv.shape[0] * hsv.shape[1]
        red_pixels = np.sum(red_mask > 0)
        green_pixels = np.sum(green_mask > 0)
        
        red_percentage = red_pixels / total_pixels
        green_percentage = green_pixels / total_pixels
        
        # Determine team based on dominant color
        min_threshold = 0.1  # Minimum 10% of jersey should be team color
        
        if red_percentage > green_percentage and red_percentage > min_threshold:
            return "red", red_percentage
        elif green_percentage > red_percentage and green_percentage > min_threshold:
            return "green", green_percentage
        else:
            return "unknown", max(red_percentage, green_percentage)
    
    def update_team_history(self, track_id: int, team: str):
        """Update team assignment history for temporal smoothing"""
        if track_id not in self.team_history:
            self.team_history[track_id] = []
        
        self.team_history[track_id].append(team)
        
        # Keep only recent history
        if len(self.team_history[track_id]) > self.history_length:
            self.team_history[track_id] = self.team_history[track_id][-self.history_length:]
    
    def get_stable_team_assignment(self, track_id: int) -> str:
        """Get stable team assignment using temporal smoothing"""
        if track_id not in self.team_history:
            return "unknown"
        
        history = self.team_history[track_id]
        if len(history) < 3:  # Need at least 3 observations
            return "unknown"
        
        # Use majority vote from recent history
        team_counts = Counter(history)
        most_common_team, count = team_counts.most_common(1)[0]
        
        # Require at least 60% agreement for stable assignment
        if count / len(history) >= 0.6:
            return most_common_team
        else:
            return "unknown"
    
    def classify_teams(self, image: np.ndarray, tracked_objects: List[Dict], 
                      detections: List[Dict]) -> List[Dict]:
        """
        Classify all tracked players into teams
        
        Args:
            image: Full frame image
            tracked_objects: List of tracked objects from DeepSORT
            detections: Original detections with masks
            
        Returns:
            List of tracked objects with team assignments
        """
        # Create mapping from bbox to mask
        bbox_to_mask = {}
        for det in detections:
            if 'mask' in det:
                bbox_key = tuple(det['bbox'])
                bbox_to_mask[bbox_key] = det['mask']
        
        classified_objects = []
        
        for tracked_obj in tracked_objects:
            track_id = tracked_obj['track_id']
            bbox = tracked_obj['bbox']
            
            # Get mask if available
            bbox_key = tuple(bbox)
            mask = bbox_to_mask.get(bbox_key, None)
            
            # Extract jersey region
            jersey_region = self.extract_jersey_region(image, bbox, mask)
            
            # Classify jersey color
            team, confidence = self.classify_jersey_color(jersey_region)
            
            # Update history
            self.update_team_history(track_id, team)
            
            # Get stable team assignment
            stable_team = self.get_stable_team_assignment(track_id)
            
            # Add team info to tracked object
            classified_obj = tracked_obj.copy()
            classified_obj.update({
                'team': stable_team,
                'team_confidence': confidence,
                'raw_team': team  # Current frame classification
            })
            
            classified_objects.append(classified_obj)
        
        return classified_objects
```

### Step 4.2: Update Main Pipeline with Team Classification

**File**: Update `src/pipeline/basketball_tracker.py` (add team classification)

```python
# Add to the BasketballTracker.__init__ method:
self.team_classifier = TeamClassifier()

# Update the process_frame method:
def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
    """Process single frame through complete pipeline with team classification"""
    # Step 1: Detect players with YOLO
    detections = self.detector.detect_players(frame)
    
    # Step 2: Segment players with SAM2
    if detections:
        segmented_detections = self.segmenter.segment_players(frame, detections)
    else:
        segmented_detections = []
    
    # Step 3: Track players with DeepSORT
    tracked_objects = self.tracker.update(segmented_detections, frame)
    
    # Step 4: Classify teams
    classified_objects = self.team_classifier.classify_teams(frame, tracked_objects, segmented_detections)
    
    # Step 5: Visualize results
    vis_frame = self.visualize_tracking_with_teams(frame, classified_objects, segmented_detections)
    
    return vis_frame, classified_objects

# Add new visualization method:
def visualize_tracking_with_teams(self, frame: np.ndarray, tracked_objects: List[Dict], 
                                 detections: List[Dict]) -> np.ndarray:
    """Visualize tracking results with team colors"""
    vis_frame = frame.copy()
    
    # Team colors
    team_colors = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'unknown': (128, 128, 128)
    }
    
    # Create a mapping from bbox to mask
    bbox_to_mask = {}
    for det in detections:
        if 'mask' in det:
            bbox_key = tuple(det['bbox'])
            bbox_to_mask[bbox_key] = det['mask']
    
    for tracked_obj in tracked_objects:
        track_id = tracked_obj['track_id']
        bbox = tracked_obj['bbox']
        team = tracked_obj.get('team', 'unknown')
        team_conf = tracked_obj.get('team_confidence', 0.0)
        
        x1, y1, x2, y2 = bbox
        
        # Get team color
        color = team_colors[team]
        
        # Draw segmentation mask if available
        bbox_key = tuple(bbox)
        if bbox_key in bbox_to_mask:
            mask = bbox_to_mask[bbox_key]
            colored_mask = np.zeros_like(vis_frame)
            colored_mask[mask] = color
            vis_frame = cv2.addWeighted(vis_frame, 0.8, colored_mask, 0.3, 0)
        
        # Draw bounding box with team color
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 3)
        
        # Draw track ID and team
        label = f"#{track_id} ({team.upper()})"
        if team_conf > 0:
            label += f" {team_conf:.2f}"
        
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Draw label background
        cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(vis_frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw center point
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(vis_frame, (center_x, center_y), 5, color, -1)
    
    # Add team statistics
    red_count = sum(1 for obj in tracked_objects if obj.get('team') == 'red')
    green_count = sum(1 for obj in tracked_objects if obj.get('team') == 'green')
    unknown_count = sum(1 for obj in tracked_objects if obj.get('team') == 'unknown')
    
    stats_text = f"Red: {red_count} | Green: {green_count} | Unknown: {unknown_count}"
    cv2.putText(vis_frame, stats_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return vis_frame
```

### Step 4.3: Test Team Classification

**File**: `scripts/test_team_classification.py`

```python
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.basketball_tracker import BasketballTracker

def test_team_classification(video_path: str):
    """Test complete pipeline with team classification"""
    tracker = BasketballTracker()
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    team_stats = {'red': 0, 'green': 0, 'unknown': 0}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        vis_frame, tracked_objects = tracker.process_frame(frame)
        
        # Update team statistics
        for obj in tracked_objects:
            team = obj.get('team', 'unknown')
            team_stats[team] += 1
        
        # Display
        cv2.imshow('Basketball Team Classification', vis_frame)
        
        # Print periodic updates
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}:")
            print(f"  Red team: {len([o for o in tracked_objects if o.get('team') == 'red'])}")
            print(f"  Green team: {len([o for o in tracked_objects if o.get('team') == 'green'])}")
            print(f"  Unknown: {len([o for o in tracked_objects if o.get('team') == 'unknown'])}")
        
        frame_count += 1
        
        if cv2.waitKey(1) & 0