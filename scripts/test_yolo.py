#!/usr/bin/env python3
"""Test YOLO Detection for Basketball Players

This script tests the YOLO detector on video input (webcam or video file)
to validate Phase 1 implementation.
"""

import cv2
import sys
import os
import time
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.detection.yolo_detector import YOLODetector


def test_yolo_on_video(video_path: str, config_path: str = None, save_output: bool = False):
    """Test YOLO detection on video"""
    print("🏀 Testing YOLO Basketball Player Detection")
    print("=" * 50)
    
    # Initialize detector
    detector = YOLODetector(
        model_path="yolo11m.pt",
        config_path=config_path or "config/yolo_config.yaml"
    )
    
    # Open video source
    if video_path.isdigit():
        cap = cv2.VideoCapture(int(video_path))
        print(f"Using webcam {video_path}")
    else:
        cap = cv2.VideoCapture(video_path)
        print(f"Using video file: {video_path}")
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {width}x{height} @ {fps} FPS")
    
    # Setup video writer if saving output
    out = None
    if save_output:
        output_path = f"output_yolo_test_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving output to: {output_path}")
    
    # Processing statistics
    frame_count = 0
    total_detections = 0
    processing_times = []
    detection_stats = {
        'total_frames': 0,
        'frames_with_detections': 0,
        'max_players_in_frame': 0,
        'confidence_scores': []
    }
    
    print("\n🎯 Starting detection... Press 'q' to quit, 's' to save frame")
    print("Keys: 'p' = pause/resume, 'r' = reset stats, 'h' = help")
    
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video or camera disconnected")
                    break
                
                # Record processing time
                start_time = time.time()
                
                # Detect players
                detections = detector.detect_players(frame)
                
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                # Update statistics
                detection_stats['total_frames'] += 1
                if detections:
                    detection_stats['frames_with_detections'] += 1
                    detection_stats['max_players_in_frame'] = max(
                        detection_stats['max_players_in_frame'], 
                        len(detections)
                    )
                    detection_stats['confidence_scores'].extend([d['confidence'] for d in detections])
                
                total_detections += len(detections)
                
                # Visualize detections
                vis_frame = detector.visualize_detections(frame, detections)
                
                # Add performance info
                fps_current = 1.0 / processing_time if processing_time > 0 else 0
                perf_text = f"FPS: {fps_current:.1f} | Processing: {processing_time*1000:.1f}ms"
                cv2.putText(vis_frame, perf_text, (10, height - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                frame_count += 1
            else:
                # Use last frame when paused
                vis_frame = detector.visualize_detections(frame, detections)
                cv2.putText(vis_frame, "PAUSED - Press 'p' to resume", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display frame
            cv2.imshow('YOLO Basketball Detection', vis_frame)
            
            # Save frame if requested
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
                save_path = f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(save_path, vis_frame)
                print(f"Frame saved to: {save_path}")
            elif key == ord('r'):
                # Reset statistics
                frame_count = 0
                total_detections = 0
                processing_times = []
                detection_stats = {
                    'total_frames': 0,
                    'frames_with_detections': 0,
                    'max_players_in_frame': 0,
                    'confidence_scores': []
                }
                print("Statistics reset")
            elif key == ord('h'):
                print_help()
            
            # Print periodic statistics
            if not paused and frame_count % 30 == 0 and frame_count > 0:
                avg_detections = total_detections / frame_count
                avg_processing_time = sum(processing_times) / len(processing_times)
                avg_fps = 1.0 / avg_processing_time
                
                print(f"Frame {frame_count}: {len(detections)} players | "
                      f"Avg: {avg_detections:.1f} players/frame | "
                      f"FPS: {avg_fps:.1f}")
                
                if detections:
                    det_stats = detector.get_detection_statistics(detections)
                    print(f"  Confidence - Avg: {det_stats['avg_confidence']:.2f}, "
                          f"Max: {det_stats['max_confidence']:.2f}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print_final_statistics(detection_stats, processing_times, frame_count)


def print_help():
    """Print help information"""
    print("\n" + "="*50)
    print("YOLO DETECTION TEST - KEYBOARD CONTROLS")
    print("="*50)
    print("q - Quit application")
    print("p - Pause/Resume processing")
    print("s - Save current frame")
    print("r - Reset statistics")
    print("h - Show this help")
    print("="*50)


def print_final_statistics(detection_stats, processing_times, frame_count):
    """Print comprehensive final statistics"""
    print("\n" + "="*60)
    print("YOLO DETECTION TEST - FINAL RESULTS")
    print("="*60)
    
    if frame_count == 0:
        print("No frames processed")
        return
    
    # Processing performance
    avg_processing_time = sum(processing_times) / len(processing_times)
    avg_fps = 1.0 / avg_processing_time
    max_fps = 1.0 / min(processing_times)
    min_fps = 1.0 / max(processing_times)
    
    print(f"📊 Performance Metrics:")
    print(f"  Total frames processed: {frame_count}")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Max FPS: {max_fps:.2f}")
    print(f"  Min FPS: {min_fps:.2f}")
    print(f"  Average processing time: {avg_processing_time*1000:.1f} ms")
    
    # Detection statistics
    detection_rate = (detection_stats['frames_with_detections'] / detection_stats['total_frames']) * 100
    avg_players = len(detection_stats['confidence_scores']) / detection_stats['total_frames']
    
    print(f"\n🎯 Detection Metrics:")
    print(f"  Frames with detections: {detection_stats['frames_with_detections']}/{detection_stats['total_frames']} ({detection_rate:.1f}%)")
    print(f"  Average players per frame: {avg_players:.2f}")
    print(f"  Maximum players in single frame: {detection_stats['max_players_in_frame']}")
    
    if detection_stats['confidence_scores']:
        import numpy as np
        confidences = detection_stats['confidence_scores']
        print(f"  Average confidence: {np.mean(confidences):.3f}")
        print(f"  Confidence range: {np.min(confidences):.3f} - {np.max(confidences):.3f}")
        
        # Confidence distribution
        high_conf = sum(1 for c in confidences if c > 0.8)
        med_conf = sum(1 for c in confidences if 0.5 < c <= 0.8)
        low_conf = sum(1 for c in confidences if c <= 0.5)
        total_conf = len(confidences)
        
        print(f"  Confidence distribution:")
        print(f"    High (>0.8): {high_conf}/{total_conf} ({100*high_conf/total_conf:.1f}%)")
        print(f"    Medium (0.5-0.8): {med_conf}/{total_conf} ({100*med_conf/total_conf:.1f}%)")
        print(f"    Low (<0.5): {low_conf}/{total_conf} ({100*low_conf/total_conf:.1f}%)")
    
    # Performance assessment
    print(f"\n✅ Phase 1 Assessment:")
    if avg_fps >= 15:
        print(f"  ✅ Performance: GOOD ({avg_fps:.1f} FPS >= 15 FPS target)")
    elif avg_fps >= 10:
        print(f"  ⚠️  Performance: ACCEPTABLE ({avg_fps:.1f} FPS)")
    else:
        print(f"  ❌ Performance: NEEDS IMPROVEMENT ({avg_fps:.1f} FPS < 10 FPS)")
    
    if detection_rate >= 80:
        print(f"  ✅ Detection Rate: GOOD ({detection_rate:.1f}% >= 80%)")
    elif detection_rate >= 60:
        print(f"  ⚠️  Detection Rate: ACCEPTABLE ({detection_rate:.1f}%)")
    else:
        print(f"  ❌ Detection Rate: NEEDS IMPROVEMENT ({detection_rate:.1f}% < 60%)")
    
    if detection_stats['confidence_scores']:
        avg_conf = np.mean(detection_stats['confidence_scores'])
        if avg_conf >= 0.7:
            print(f"  ✅ Confidence: GOOD ({avg_conf:.3f} >= 0.7)")
        elif avg_conf >= 0.5:
            print(f"  ⚠️  Confidence: ACCEPTABLE ({avg_conf:.3f})")
        else:
            print(f"  ❌ Confidence: NEEDS IMPROVEMENT ({avg_conf:.3f} < 0.5)")


def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Test YOLO Basketball Player Detection')
    parser.add_argument('--input', '-i', type=str, default='0',
                       help='Input source (0 for webcam, path for video file)')
    parser.add_argument('--config', '-c', type=str, default=None,
                       help='Path to YOLO configuration file')
    parser.add_argument('--save', '-s', action='store_true',
                       help='Save output video')
    
    args = parser.parse_args()
    
    # Test YOLO detection
    test_yolo_on_video(args.input, args.config, args.save)


if __name__ == "__main__":
    main()
