#!/usr/bin/env python3
"""Process Short Video Segment and Playback

This script processes a short segment (10 seconds) of the video for quick testing.
"""

import cv2
import numpy as np
import time
import argparse
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline.basketball_tracker import BasketballTracker

# ByteTrack is optional
try:
    from src.pipeline.basketball_tracker_bytetrack import BasketballTrackerByteTrack
    BYTETRACK_AVAILABLE = True
except ImportError:
    BYTETRACK_AVAILABLE = False


def process_short_segment(
    input_path: str,
    output_path: str,
    start_time: float = 0,
    duration: float = 10,
    optimization_level: str = "balanced",
    target_fps: int = 30,
    method: str = "deepsort",
    mask_only: bool = False
):
    """
    Process a short segment of video and save results
    
    Args:
        input_path: Path to input video
        output_path: Path to save processed video
        start_time: Start time in seconds
        duration: Duration to process in seconds
        optimization_level: Pipeline optimization level
        target_fps: Target FPS for output video
        method: Tracking method (deepsort or bytetrack)
        mask_only: Show only segmentation masks without bounding boxes
    """
    print("🏀 Basketball Short Segment Processing")
    print("=" * 50)
    print(f"📹 Input: {input_path}")
    print(f"💾 Output: {output_path}")
    print(f"⏰ Segment: {start_time}s - {start_time + duration}s ({duration}s)")
    print(f"🎯 Optimization: {optimization_level}")
    print(f"🎬 Target FPS: {target_fps}")
    print(f"🔧 Method: {method.upper()}")
    print(f"🎭 Mask-only mode: {mask_only}")
    print()
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {input_path}")
        return False
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate frame range
    start_frame = int(start_time * original_fps)
    end_frame = int((start_time + duration) * original_fps)
    total_frames = end_frame - start_frame
    
    print(f"📊 Segment Info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  Original FPS: {original_fps:.1f}")
    print(f"  Processing frames: {start_frame} - {end_frame} ({total_frames} frames)")
    print()
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Initialize basketball tracker based on method
    print("🔧 Initializing basketball tracking pipeline...")
    if method.lower() == "bytetrack":
        if not BYTETRACK_AVAILABLE:
            print("❌ ByteTrack not available. Install with: pip install supervision")
            print("Falling back to DeepSORT...")
            tracker = BasketballTracker(optimization_level=optimization_level)
        else:
            tracker = BasketballTrackerByteTrack(
                optimization_level=optimization_level,
                mask_only_visualization=mask_only
            )
    else:  # deepsort (default)
        tracker = BasketballTracker(optimization_level=optimization_level)
    print()
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
    
    # Process frames
    print("🎬 Processing frames...")
    print("Progress: [" + " " * 40 + "] 0%", end="\r")
    
    frame_count = 0
    start_time_proc = time.time()
    
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame through pipeline
        vis_frame, tracked_players = tracker.process_frame(frame)
        
        # Write frame to output video
        out.write(vis_frame)
        
        frame_count += 1
        
        # Update progress
        progress = frame_count / total_frames
        bar_length = int(progress * 40)
        bar = "█" * bar_length + " " * (40 - bar_length)
        print(f"Progress: [{bar}] {progress*100:.1f}% ({frame_count}/{total_frames})", end="\r")
    
    # Cleanup
    cap.release()
    out.release()
    
    processing_time = time.time() - start_time_proc
    processing_fps = frame_count / processing_time
    
    print(f"\n\n✅ Processing Complete!")
    print(f"📊 Processing Stats:")
    print(f"  Processed frames: {frame_count}")
    print(f"  Processing time: {processing_time:.1f} seconds")
    print(f"  Processing FPS: {processing_fps:.2f}")
    print(f"  Speed ratio: {processing_fps/original_fps:.2f}x")
    print(f"💾 Output saved: {output_path}")
    
    return True


def playback_video(video_path: str, target_fps: int = 60):
    """Play processed video at target FPS"""
    print(f"\n🎬 Playing back video at {target_fps} FPS")
    print("Controls: 'q' to quit, 'p' or SPACE to pause/resume, 'r' to restart")
    print("=" * 50)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open processed video {video_path}")
        return
    
    frame_delay = 1.0 / target_fps
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # Loop back to beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
        
        # Display frame
        cv2.imshow("Basketball Tracking - Short Segment", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(int(frame_delay * 1000) if not paused else 0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('p') or key == ord(' '):
            paused = not paused
            print(f"{'⏸️  Paused' if paused else '▶️  Playing'}")
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            paused = False
            print("🔄 Restarted")
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Process short basketball video segment")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", help="Output video path (default: auto-generated)")
    parser.add_argument("--start", "-s", type=float, default=0, help="Start time in seconds (default: 0)")
    parser.add_argument("--duration", "-d", type=float, default=10, help="Duration in seconds (default: 10)")
    parser.add_argument("--optimization", "-opt", default="balanced", 
                       choices=["fast", "balanced", "quality"],
                       help="Pipeline optimization level")
    parser.add_argument("--fps", "-f", type=int, default=30, 
                       help="Target playback FPS (default: 30)")
    parser.add_argument("--method", "-m", default="deepsort",
                       choices=["deepsort", "bytetrack"],
                       help="Tracking method (default: deepsort)")
    parser.add_argument("--mask-only", action="store_true",
                       help="Show only segmentation masks without bounding boxes (ByteTrack only)")
    parser.add_argument("--no-playback", action="store_true",
                       help="Skip playback, only process and save")
    
    args = parser.parse_args()
    
    # Generate output path if not provided
    if not args.output:
        input_path = Path(args.input)
        mask_suffix = "_masks" if args.mask_only else ""
        args.output = str(input_path.parent / f"{input_path.stem}_short_{args.start}s_{args.duration}s_{args.optimization}_{args.method}{mask_suffix}.mp4")
    
    try:
        # Process video segment
        success = process_short_segment(
            args.input, 
            args.output, 
            args.start,
            args.duration,
            args.optimization,
            args.fps,
            args.method,
            args.mask_only
        )
        
        if success and not args.no_playback:
            # Playback processed video
            playback_video(args.output, args.fps)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
