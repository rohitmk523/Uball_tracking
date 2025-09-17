#!/usr/bin/env python3
"""Test YOLO + SAM2 Pipeline for Basketball Players

This script tests the complete YOLO + SAM2 pipeline for basketball player
detection and segmentation with comprehensive evaluation and controls.
"""

import cv2
import sys
import os
import time
import argparse
import json
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline.yolo_sam2_pipeline import YOLOSAMPipeline


def test_yolo_sam2_pipeline(
    video_path: str,
    optimization_level: str = "balanced",
    save_output: bool = False,
    output_path: str = None,
    save_stats: bool = False
):
    """Test YOLO + SAM2 pipeline on video input"""
    print("🏀 Testing YOLO + SAM2 Basketball Player Pipeline")
    print("=" * 60)
    
    # Initialize pipeline
    print(f"🔧 Initializing pipeline (optimization: {optimization_level})...")
    try:
        pipeline = YOLOSAMPipeline(
            yolo_config_path="config/yolo_config.yaml",
            sam2_config_path="config/sam2_config.yaml",
            optimization_level=optimization_level
        )
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return False
    
    # Open video source
    if video_path.isdigit():
        cap = cv2.VideoCapture(int(video_path))
        source_name = f"webcam {video_path}"
        is_webcam = True
    else:
        cap = cv2.VideoCapture(video_path)
        source_name = f"video file: {video_path}"
        is_webcam = False
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open {source_name}")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else -1
    
    print(f"📹 Video properties: {width}x{height} @ {fps} FPS")
    if not is_webcam:
        print(f"📊 Total frames: {total_frames}")
    
    # Setup output video writer if requested
    out = None
    if save_output:
        if not output_path:
            timestamp = int(time.time())
            output_path = f"output_yolo_sam2_{optimization_level}_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"💾 Recording to: {output_path}")
    
    # Processing statistics
    frame_count = 0
    all_frame_data = []
    start_time = time.time()
    
    # Display settings
    display_settings = {
        'show_masks': True,
        'show_boxes': True,
        'show_scores': True,
        'mask_alpha': 0.4
    }
    
    print("\n🎬 Starting processing...")
    print("Controls:")
    print("  'q' - Quit")
    print("  'p' - Pause/Resume")
    print("  's' - Save current frame")
    print("  'm' - Toggle mask display")
    print("  'b' - Toggle box display")
    print("  'r' - Reset statistics")
    print("  'h' - Show help")
    print("  '+'/'-' - Adjust mask transparency")
    print("-" * 60)
    
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if is_webcam:
                        print("Camera disconnected")
                    else:
                        print("End of video")
                    break
                
                # Process frame through pipeline
                vis_frame, detections = pipeline.process_frame(frame)
                
                # Store frame data for statistics
                frame_data = {
                    'frame': frame_count,
                    'detections': len(detections),
                    'segmented': sum(1 for d in detections if d.get('has_mask', False)),
                    'timestamp': time.time()
                }
                all_frame_data.append(frame_data)
                
                frame_count += 1
            else:
                # Show paused frame
                pause_text = "PAUSED - Press 'p' to resume"
                cv2.putText(vis_frame, pause_text, (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Apply display settings
            if 'vis_frame' in locals():
                pipeline.update_visualization_settings(
                    show_masks=display_settings['show_masks'],
                    show_boxes=display_settings['show_boxes'],
                    mask_alpha=display_settings['mask_alpha']
                )
            
            # Display frame
            cv2.imshow('YOLO + SAM2 Basketball Tracking', vis_frame)
            
            # Save frame if recording
            if out and not paused:
                out.write(vis_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quit requested by user")
                break
            elif key == ord('p'):
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
            elif key == ord('s') and not paused:
                # Save current frame
                save_frame_path = f"frame_{frame_count:06d}_yolo_sam2.jpg"
                cv2.imwrite(save_frame_path, vis_frame)
                print(f"Frame saved to: {save_frame_path}")
            elif key == ord('m'):
                display_settings['show_masks'] = not display_settings['show_masks']
                print(f"Mask display: {'ON' if display_settings['show_masks'] else 'OFF'}")
            elif key == ord('b'):
                display_settings['show_boxes'] = not display_settings['show_boxes']
                print(f"Box display: {'ON' if display_settings['show_boxes'] else 'OFF'}")
            elif key == ord('r'):
                # Reset statistics
                all_frame_data = []
                frame_count = 0
                start_time = time.time()
                print("Statistics reset")
            elif key == ord('h'):
                print_help()
            elif key == ord('+') or key == ord('='):
                display_settings['mask_alpha'] = min(1.0, display_settings['mask_alpha'] + 0.1)
                print(f"Mask alpha: {display_settings['mask_alpha']:.1f}")
            elif key == ord('-'):
                display_settings['mask_alpha'] = max(0.0, display_settings['mask_alpha'] - 0.1)
                print(f"Mask alpha: {display_settings['mask_alpha']:.1f}")
            
            # Print periodic statistics
            if not paused and frame_count % 30 == 0 and frame_count > 0:
                current_perf = pipeline.get_current_performance()
                print(f"Frame {frame_count}: {current_perf['current_fps']:.1f} FPS | "
                      f"Players: {detections[-1] if all_frame_data else 0} detected")
                
                if current_perf['sam2_available']:
                    print(f"  YOLO: {current_perf['avg_detection_time_ms']:.1f}ms | "
                          f"SAM2: {current_perf['avg_segmentation_time_ms']:.1f}ms")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Calculate and display final statistics
        total_time = time.time() - start_time
        print_final_statistics(pipeline, all_frame_data, total_time, optimization_level)
        
        # Save statistics if requested
        if save_stats:
            save_performance_statistics(pipeline, all_frame_data, optimization_level)
    
    return True


def print_help():
    """Print help information"""
    print("\n" + "="*60)
    print("YOLO + SAM2 PIPELINE TEST - KEYBOARD CONTROLS")
    print("="*60)
    print("q - Quit application")
    print("p - Pause/Resume processing")
    print("s - Save current frame")
    print("m - Toggle mask display")
    print("b - Toggle bounding box display") 
    print("r - Reset statistics")
    print("h - Show this help")
    print("+ - Increase mask transparency")
    print("- - Decrease mask transparency")
    print("="*60)


def print_final_statistics(pipeline, frame_data, total_time, optimization_level):
    """Print comprehensive final statistics"""
    print("\n" + "="*70)
    print("YOLO + SAM2 PIPELINE TEST - FINAL RESULTS")
    print("="*70)
    
    if not frame_data:
        print("No frames processed")
        return
    
    # Basic statistics
    total_frames = len(frame_data)
    total_detections = sum(fd['detections'] for fd in frame_data)
    total_segmented = sum(fd['segmented'] for fd in frame_data)
    avg_fps = total_frames / total_time if total_time > 0 else 0
    
    print(f"📊 Processing Summary:")
    print(f"  Total frames processed: {total_frames}")
    print(f"  Total processing time: {total_time:.2f} seconds")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Optimization level: {optimization_level}")
    
    print(f"\n🎯 Detection Results:")
    print(f"  Total player detections: {total_detections}")
    print(f"  Average players per frame: {total_detections / total_frames:.2f}")
    print(f"  Frames with players: {sum(1 for fd in frame_data if fd['detections'] > 0)}")
    print(f"  Max players in single frame: {max(fd['detections'] for fd in frame_data)}")
    
    print(f"\n🖼️  Segmentation Results:")
    print(f"  Total successful segmentations: {total_segmented}")
    print(f"  Segmentation success rate: {(total_segmented / total_detections * 100) if total_detections > 0 else 0:.1f}%")
    print(f"  Average segmented per frame: {total_segmented / total_frames:.2f}")
    print(f"  SAM2 status: {'Available' if pipeline.sam2_available else 'Not Available (using fallback)'}")
    
    # Performance assessment
    print(f"\n✅ Phase 2 Assessment:")
    
    # FPS assessment
    if avg_fps >= 10:
        print(f"  ✅ Performance: GOOD ({avg_fps:.1f} FPS >= 10 FPS target)")
    elif avg_fps >= 5:
        print(f"  ⚠️  Performance: ACCEPTABLE ({avg_fps:.1f} FPS)")
    else:
        print(f"  ❌ Performance: NEEDS IMPROVEMENT ({avg_fps:.1f} FPS < 5 FPS)")
    
    # Detection rate assessment
    detection_rate = (sum(1 for fd in frame_data if fd['detections'] > 0) / total_frames) * 100
    if detection_rate >= 80:
        print(f"  ✅ Detection Rate: GOOD ({detection_rate:.1f}% frames with players)")
    elif detection_rate >= 60:
        print(f"  ⚠️  Detection Rate: ACCEPTABLE ({detection_rate:.1f}%)")
    else:
        print(f"  ❌ Detection Rate: NEEDS IMPROVEMENT ({detection_rate:.1f}%)")
    
    # Segmentation assessment
    if total_detections > 0:
        seg_rate = (total_segmented / total_detections) * 100
        if seg_rate >= 90:
            print(f"  ✅ Segmentation: EXCELLENT ({seg_rate:.1f}% success rate)")
        elif seg_rate >= 70:
            print(f"  ✅ Segmentation: GOOD ({seg_rate:.1f}% success rate)")
        elif seg_rate >= 50:
            print(f"  ⚠️  Segmentation: ACCEPTABLE ({seg_rate:.1f}% success rate)")
        else:
            print(f"  ❌ Segmentation: NEEDS IMPROVEMENT ({seg_rate:.1f}% success rate)")
    
    # Get detailed performance metrics
    current_perf = pipeline.get_current_performance()
    if current_perf.get('avg_detection_time_ms', 0) > 0:
        print(f"\n⚡ Component Performance:")
        print(f"  YOLO Detection: {current_perf['avg_detection_time_ms']:.1f} ms/frame")
        print(f"  SAM2 Segmentation: {current_perf['avg_segmentation_time_ms']:.1f} ms/frame")


def save_performance_statistics(pipeline, frame_data, optimization_level):
    """Save detailed performance statistics to JSON file"""
    timestamp = int(time.time())
    filename = f"yolo_sam2_stats_{optimization_level}_{timestamp}.json"
    
    # Prepare statistics data
    stats_data = {
        'test_info': {
            'timestamp': timestamp,
            'optimization_level': optimization_level,
            'sam2_available': pipeline.sam2_available,
            'total_frames': len(frame_data)
        },
        'performance_metrics': pipeline.get_current_performance(),
        'frame_by_frame_data': frame_data,
        'summary': {
            'total_detections': sum(fd['detections'] for fd in frame_data),
            'total_segmented': sum(fd['segmented'] for fd in frame_data),
            'avg_detections_per_frame': sum(fd['detections'] for fd in frame_data) / len(frame_data) if frame_data else 0,
            'avg_segmented_per_frame': sum(fd['segmented'] for fd in frame_data) / len(frame_data) if frame_data else 0
        }
    }
    
    try:
        with open(filename, 'w') as f:
            json.dump(stats_data, f, indent=2)
        print(f"📊 Performance statistics saved to: {filename}")
    except Exception as e:
        print(f"❌ Failed to save statistics: {e}")


def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Test YOLO + SAM2 Basketball Player Pipeline')
    parser.add_argument('--input', '-i', type=str, default='0',
                       help='Input source (0 for webcam, path for video file)')
    parser.add_argument('--optimization', '-opt', choices=['fast', 'balanced', 'quality'], 
                       default='balanced', help='Optimization level')
    parser.add_argument('--save-output', '-so', action='store_true',
                       help='Save output video')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output video file path')
    parser.add_argument('--save-stats', '-ss', action='store_true',
                       help='Save performance statistics')
    
    args = parser.parse_args()
    
    # Test YOLO + SAM2 pipeline
    success = test_yolo_sam2_pipeline(
        video_path=args.input,
        optimization_level=args.optimization,
        save_output=args.save_output,
        output_path=args.output,
        save_stats=args.save_stats
    )
    
    if success:
        print("\n🎉 Phase 2 testing completed successfully!")
        print("\nNext steps:")
        print("1. Review segmentation quality and performance")
        print("2. Tune SAM2 configuration if needed")
        print("3. Ready for Phase 3: DeepSORT tracking integration")
    else:
        print("\n❌ Phase 2 testing encountered issues")
        print("Please check the error messages above and resolve any problems")


if __name__ == "__main__":
    main()
