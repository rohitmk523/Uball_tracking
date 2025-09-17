# 🎬 Basketball Video Processing Guide

Complete guide for processing basketball videos using the YOLO11 + SAM2 + ByteTrack pipeline.

## 📋 Table of Contents

1. [Quick Start](#-quick-start)
2. [Short Segment Processing](#-short-segment-processing)
3. [Full Video Processing](#-full-video-processing)
4. [Optimization Levels](#️-optimization-levels)
5. [Tracking Methods](#-tracking-methods)
6. [Output Files](#-output-files)
7. [Performance Guide](#-performance-guide)
8. [Troubleshooting](#-troubleshooting)

## 🚀 Quick Start

### Prerequisites
```bash
# Ensure environment is set up
conda activate basketball_tracking

# Download models if not already done
python scripts/download_models.py
python scripts/download_sam2_models.py  # Choose option 1 or 2 for faster processing
```

### Test Your Setup
```bash
# Quick test (5 seconds)
python scripts/process_short_segment.py \
    --input your_video.mp4 \
    --start 10 \
    --duration 5 \
    --optimization fast \
    --method bytetrack
```

## 🎯 Short Segment Processing

Perfect for testing, experimentation, and quick analysis of specific game moments.

### Basic Usage

```bash
python scripts/process_short_segment.py [OPTIONS]
```

### Command Line Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--input, -i` | Input video path | **Required** | `video.mp4` |
| `--start, -s` | Start time (seconds) | 0 | `30` |
| `--duration, -d` | Duration (seconds) | 10 | `15` |
| `--optimization, -opt` | Optimization level | balanced | `fast/balanced/quality` |
| `--method, -m` | Tracking method | bytetrack | `bytetrack/deepsort` |
| `--output, -o` | Output path | Auto-generated | `output.mp4` |

### Example Commands

```bash
# 🏃‍♂️ Fast testing (no SAM2, quick results)
python scripts/process_short_segment.py \
    --input game_footage.mp4 \
    --start 120 \
    --duration 8 \
    --optimization fast \
    --method bytetrack

# ⚖️ Balanced processing (recommended for most use cases)
python scripts/process_short_segment.py \
    --input game_footage.mp4 \
    --start 45 \
    --duration 12 \
    --optimization balanced \
    --method bytetrack

# 🎯 High quality (best results, slower)
python scripts/process_short_segment.py \
    --input game_footage.mp4 \
    --start 200 \
    --duration 20 \
    --optimization quality \
    --method bytetrack

# 🔄 Compare methods (process same segment with both)
python scripts/process_short_segment.py --input game.mp4 --start 60 --duration 10 --method bytetrack
python scripts/process_short_segment.py --input game.mp4 --start 60 --duration 10 --method deepsort
```

### Output Naming

Short segment outputs are automatically named:
```
{input_name}_short_{start}s_{duration}s_{optimization}_{method}.mp4
```

Example: `game_footage_short_45.0s_12.0s_balanced_bytetrack.mp4`

## 🎬 Full Video Processing

Process entire videos with offline rendering and optional playback.

### Basic Usage

```bash
python scripts/process_and_playback.py [OPTIONS]
```

### Command Line Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--input, -i` | Input video path | **Required** | `full_game.mp4` |
| `--optimization, -opt` | Optimization level | balanced | `fast/balanced/quality` |
| `--method, -m` | Tracking method | deepsort | `bytetrack/deepsort` |
| `--fps, -f` | Output FPS | 30 | `30` |
| `--output, -o` | Output path | Auto-generated | `tracked_game.mp4` |
| `--no-playback` | Skip playback after processing | False | Flag only |
| `--playback-only` | Only playback existing video | None | `existing.mp4` |

### Example Commands

```bash
# 🎬 Process entire game with recommended settings
python scripts/process_and_playback.py \
    --input full_game.mp4 \
    --optimization balanced \
    --method bytetrack

# 🚀 Fast processing for quick preview
python scripts/process_and_playback.py \
    --input full_game.mp4 \
    --optimization fast \
    --method bytetrack

# 💾 Process without playback (save processing time)
python scripts/process_and_playback.py \
    --input full_game.mp4 \
    --optimization balanced \
    --method bytetrack \
    --no-playback

# 📺 Playback previously processed video
python scripts/process_and_playback.py \
    --playback-only full_game_tracked_balanced_bytetrack_30fps.mp4

# 🎯 High quality for final analysis
python scripts/process_and_playback.py \
    --input championship_game.mp4 \
    --optimization quality \
    --method bytetrack \
    --no-playback
```

### Output Naming

Full video outputs are automatically named:
```
{input_name}_tracked_{optimization}_{method}_{fps}fps.mp4
```

Example: `championship_game_tracked_balanced_bytetrack_30fps.mp4`

## ⚙️ Optimization Levels

Choose the right balance of speed, quality, and resource usage.

### 🏃‍♂️ Fast Mode
- **Models**: YOLO11s, No SAM2
- **Speed**: ~15-20 FPS processing
- **Quality**: Good detection, bounding boxes only
- **Use Cases**: 
  - Quick testing and iteration
  - Real-time applications
  - Resource-constrained environments
  - Initial footage review

```bash
--optimization fast
```

### ⚖️ Balanced Mode (Recommended)
- **Models**: YOLO11m, SAM2-tiny
- **Speed**: ~5-10 FPS processing  
- **Quality**: Great detection + pixel-perfect masks
- **Use Cases**:
  - Production analysis
  - Most basketball tracking tasks
  - Best speed/quality balance
  - Standard workflow

```bash
--optimization balanced
```

### 🎯 Quality Mode
- **Models**: YOLO11l, SAM2-small/base
- **Speed**: ~2-5 FPS processing
- **Quality**: Excellent detection + high-quality masks
- **Use Cases**:
  - Final production videos
  - Detailed player analysis
  - Research applications
  - Publication-quality results

```bash
--optimization quality
```

## 🔧 Tracking Methods

### 🚀 ByteTrack (Recommended)

**Pros:**
- ✅ Stable player IDs with minimal flickering
- ✅ Fast processing speed
- ✅ Excellent for basketball (handles fast movements)
- ✅ Works well with SAM2 mask features
- ✅ Robust to temporary occlusions

**Cons:**
- ❌ Primarily motion-based (less appearance info)
- ❌ May struggle with very similar players

**Best For:** Basketball tracking, sports analysis, real-time applications

```bash
--method bytetrack
```

### 🧠 DeepSORT

**Pros:**
- ✅ Uses appearance features for re-identification
- ✅ Better for long-term occlusions
- ✅ Good for general object tracking

**Cons:**
- ❌ Can have ID flickering in basketball
- ❌ Slower processing
- ❌ More sensitive to appearance changes

**Best For:** General tracking, scenarios with long occlusions

```bash
--method deepsort
```

## 📁 Output Files

### File Structure
```
your_video_directory/
├── original_video.mp4                    # Your input video
├── original_video_short_30s_10s_balanced_bytetrack.mp4    # Short segment
├── original_video_tracked_balanced_bytetrack_30fps.mp4    # Full video
└── processing_logs/                      # Debug info (if enabled)
```

### Video Features
- **Resolution**: Maintains original video resolution
- **Frame Rate**: 30 FPS (configurable)
- **Codec**: H.264 (MP4V) for broad compatibility
- **Overlays**: 
  - Colored bounding boxes per player
  - Unique player IDs
  - Pixel-perfect colored masks (balanced/quality modes)
  - Confidence scores (debug mode)

## 📊 Performance Guide

### Processing Time Estimates

For a **60-second basketball video** (1920x1080, 30fps):

| Configuration | CPU (8-core) | GPU (RTX 3070) | GPU (RTX 4090) |
|---------------|--------------|----------------|----------------|
| Fast + ByteTrack | ~8-12 min | ~3-5 min | ~2-3 min |
| Balanced + ByteTrack | ~45-60 min | ~15-25 min | ~8-12 min |
| Quality + ByteTrack | ~2-3 hours | ~45-90 min | ~25-40 min |

### Memory Requirements

| Mode | RAM | VRAM | Storage (60s video) |
|------|-----|------|-------------------|
| Fast | 4-6 GB | 2-3 GB | ~500 MB |
| Balanced | 6-8 GB | 4-6 GB | ~800 MB |
| Quality | 8-12 GB | 6-8 GB | ~1.2 GB |

### Optimization Tips

1. **GPU Acceleration**: Ensure CUDA is properly installed
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Batch Processing**: Process multiple segments in parallel
   ```bash
   # Process different segments simultaneously
   python scripts/process_short_segment.py --input game.mp4 --start 0 --duration 30 &
   python scripts/process_short_segment.py --input game.mp4 --start 30 --duration 30 &
   ```

3. **Storage**: Use SSD for faster I/O, especially for quality mode

## 🔧 Troubleshooting

### Common Issues

#### 1. Out of Memory Errors
```bash
# Solution: Use fast mode or reduce video resolution
python scripts/process_short_segment.py --optimization fast --duration 5
```

#### 2. Slow Processing
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Use fast mode for testing
--optimization fast
```

#### 3. Poor Tracking Quality
```bash
# Try different method
--method bytetrack  # Usually better for basketball

# Use higher quality mode
--optimization balanced  # or quality
```

#### 4. Video Format Issues
```bash
# Convert video to standard format
ffmpeg -i input.mov -c:v libx264 -c:a aac output.mp4
```

### Debug Mode

Enable verbose logging:
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
python scripts/process_short_segment.py --input video.mp4 --start 10 --duration 5 --optimization fast 2>&1 | tee debug.log
```

### Performance Monitoring

Monitor resource usage during processing:
```bash
# Install htop for system monitoring
htop

# Monitor GPU usage (if NVIDIA)
watch -n 1 nvidia-smi
```

## 📈 Advanced Usage

### Batch Processing Script

Create a script to process multiple segments:

```bash
#!/bin/bash
# process_highlights.sh

VIDEO="game_footage.mp4"
SEGMENTS=(
    "30 15"   # Start at 30s, duration 15s
    "120 20"  # Start at 120s, duration 20s  
    "300 25"  # Start at 300s, duration 25s
)

for segment in "${SEGMENTS[@]}"; do
    read start duration <<< "$segment"
    echo "Processing segment: ${start}s for ${duration}s"
    python scripts/process_short_segment.py \
        --input "$VIDEO" \
        --start "$start" \
        --duration "$duration" \
        --optimization balanced \
        --method bytetrack
done
```

### Configuration Files

Create custom configs for repeated processing:

```yaml
# custom_config.yaml
optimization: balanced
method: bytetrack
segments:
  - start: 45
    duration: 12
  - start: 180
    duration: 18
  - start: 420
    duration: 15
```

---

## 🎯 Quick Reference

### Most Common Commands

```bash
# Quick test
python scripts/process_short_segment.py -i video.mp4 -s 30 -d 10 -opt fast -m bytetrack

# Production segment  
python scripts/process_short_segment.py -i video.mp4 -s 120 -d 20 -opt balanced -m bytetrack

# Full video processing
python scripts/process_and_playback.py -i video.mp4 -opt balanced -m bytetrack --no-playback

# Playback existing
python scripts/process_and_playback.py --playback-only processed_video.mp4
```

### File Size Estimates

| Video Length | Resolution | Fast Mode | Balanced Mode | Quality Mode |
|--------------|------------|-----------|---------------|--------------|
| 10 seconds | 1080p | ~15 MB | ~25 MB | ~40 MB |
| 60 seconds | 1080p | ~90 MB | ~150 MB | ~240 MB |
| 10 minutes | 1080p | ~900 MB | ~1.5 GB | ~2.4 GB |

---

**Ready to track some basketball players?** 🏀 Start with a short segment test and work your way up to full video processing!
