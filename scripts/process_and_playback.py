#!/usr/bin/env python3
"""Process Video Offline and Playback at 60fps

This script processes the entire video through the basketball tracking pipeline
offline, saves the results, and then plays it back at smooth 60fps.
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


def process_video_offline(
    input_path: str,
    output_path: str,
    optimization_level: str = "balanced",
    target_fps: int = 30,
    method: str = "deepsort"
):
    """
    Process entire video offline and save results
    
    Args:
        input_path: Path to input video
        output_path: Path to save processed video
        optimization_level: Pipeline optimization level
        target_fps: Target FPS for output video
    """
    print("🏀 Basketball Video Offline Processing")
    print("=" * 60)
    print(f"📹 Input: {input_path}")
    print(f"💾 Output: {output_path}")
    print(f"🎯 Optimization: {optimization_level}")
    print(f"🎬 Target FPS: {target_fps}")
    print(f"🔧 Method: {method.upper()}")
    print()
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {input_path}")
        return False
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📊 Video Info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  Original FPS: {original_fps:.1f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/original_fps:.1f} seconds")
    print()
    
    # Initialize basketball tracker based on method
    print("🔧 Initializing basketball tracking pipeline...")
    if method.lower() == "bytetrack":
        if not BYTETRACK_AVAILABLE:
            print("❌ ByteTrack not available. Install with: pip install supervision")
            print("Falling back to DeepSORT...")
            tracker = BasketballTracker(optimization_level=optimization_level)
        else:
            tracker = BasketballTrackerByteTrack(optimization_level=optimization_level)
    else:  # deepsort (default)
        tracker = BasketballTracker(optimization_level=optimization_level)
    print()
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
    
    # Process all frames
    print("🎬 Processing frames...")
    print("Progress: [" + " " * 50 + "] 0%", end="\r")
    
    processed_frames = []
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame through pipeline
        vis_frame, tracked_players = tracker.process_frame(frame)
        processed_frames.append(vis_frame)
        
        frame_count += 1
        
        # Update progress
        progress = frame_count / total_frames
        bar_length = int(progress * 50)
        bar = "█" * bar_length + " " * (50 - bar_length)
        print(f"Progress: [{bar}] {progress*100:.1f}% ({frame_count}/{total_frames})", end="\r")
        
        # Write frame to output video
        out.write(vis_frame)
    
    # Cleanup
    cap.release()
    out.release()
    
    processing_time = time.time() - start_time
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
    """
    Play processed video at target FPS
    
    Args:
        video_path: Path to processed video
        target_fps: Playback FPS
    """
    print(f"\n🎬 Playing back video at {target_fps} FPS")
    print("Controls:")
    print("  'q' - Quit")
    print("  'p' - Pause/Resume")
    print("  'r' - Restart")
    print("  'SPACE' - Pause/Resume")
    print("  '←/→' - Skip backward/forward 5 seconds")
    print("=" * 60)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open processed video {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_delay = 1.0 / target_fps
    
    paused = False
    current_frame = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # Loop back to beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                current_frame = 0
                continue
            
            current_frame += 1
        
        # Display frame
        cv2.imshow("Basketball Tracking - Processed Video", frame)
        
        # Handle keyboard input
        if paused:
            key = cv2.waitKey(0) & 0xFF
        else:
            key = cv2.waitKey(int(frame_delay * 1000)) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('p') or key == ord(' '):
            paused = not paused
            print(f"{'⏸️  Paused' if paused else '▶️  Playing'}")
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            current_frame = 0
            paused = False
            print("🔄 Restarted")
        elif key == 81:  # Left arrow
            new_frame = max(0, current_frame - int(5 * target_fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            current_frame = new_frame
            print(f"⏪ Skipped to frame {current_frame}")
        elif key == 83:  # Right arrow
            new_frame = min(total_frames - 1, current_frame + int(5 * target_fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            current_frame = new_frame
            print(f"⏩ Skipped to frame {current_frame}")
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Process basketball video offline and playback at 60fps")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", help="Output video path (default: auto-generated)")
    parser.add_argument("--optimization", "-opt", default="balanced", 
                       choices=["fast", "balanced", "quality"],
                       help="Pipeline optimization level")
    parser.add_argument("--fps", "-f", type=int, default=30, 
                       help="Target playback FPS (default: 30)")
    parser.add_argument("--method", "-m", default="deepsort",
                       choices=["deepsort", "bytetrack"],
                       help="Tracking method (default: deepsort)")
    parser.add_argument("--no-playback", action="store_true",
                       help="Skip playback, only process and save")
    parser.add_argument("--playback-only", help="Skip processing, only playback existing video")
    
    args = parser.parse_args()
    
    # Generate output path if not provided
    if not args.output and not args.playback_only:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_tracked_{args.optimization}_{args.method}_{args.fps}fps.mp4")
    
    try:
        if args.playback_only:
            # Only playback existing video
            playback_video(args.playback_only, args.fps)
        else:
            # Process video
            success = process_video_offline(
                args.input, 
                args.output, 
                args.optimization,
                args.fps,
                args.method
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
