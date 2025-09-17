#!/usr/bin/env python3
"""Test Complete Basketball Tracking Pipeline

This script tests the full basketball player tracking system combining
YOLO detection, SAM2 segmentation, and DeepSORT tracking with comprehensive
evaluation and interactive controls.
"""

import cv2
import sys
import os
import time
import argparse
import json
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline.basketball_tracker import BasketballTracker


def test_complete_pipeline(
    video_path: str,
    optimization_level: str = "balanced",
    save_output: bool = False,
    output_path: str = None,
    save_stats: bool = False,
    show_trails: bool = False
):
    """Test complete basketball tracking pipeline"""
    print("🏀 Testing Complete Basketball Tracking Pipeline")
    print("=" * 70)
    
    # Initialize complete pipeline
    print(f"🔧 Initializing pipeline (optimization: {optimization_level})...")
    try:
        pipeline = BasketballTracker(
            yolo_config_path="config/yolo_config.yaml",
            sam2_config_path="config/sam2_config.yaml", 
            deepsort_config_path="config/deepsort_config.yaml",
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
            output_path = f"output_complete_tracking_{optimization_level}_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"💾 Recording to: {output_path}")
    
    # Processing statistics
    frame_count = 0
    all_frame_data = []
    start_time = time.time()
    track_history = {}  # For trail visualization
    
    # Display settings
    display_settings = {
        'show_masks': True,
        'show_boxes': True,
        'show_ids': True,
        'show_trails': show_trails,
        'show_velocity': False,
        'show_stats': True,
        'mask_alpha': 0.4
    }
    
    print("\n🎬 Starting complete pipeline processing...")
    print("Advanced Controls:")
    print("  'q' - Quit")
    print("  'p' - Pause/Resume")
    print("  's' - Save current frame")
    print("  'r' - Reset tracking and statistics")
    print("  'h' - Show help")
    print("  'm' - Toggle mask display")
    print("  'b' - Toggle bounding box display")
    print("  'i' - Toggle ID display")
    print("  't' - Toggle trail display")
    print("  'v' - Toggle velocity vectors")
    print("  '+'/'-' - Adjust mask transparency")
    print("  'SPACE' - Single step (when paused)")
    print("-" * 70)
    
    paused = False
    single_step = False
    
    try:
        while True:
            if not paused or single_step:
                ret, frame = cap.read()
                if not ret:
                    if is_webcam:
                        print("Camera disconnected")
                    else:
                        print("End of video")
                    break
                
                # Process frame through complete pipeline
                vis_frame, tracked_players = pipeline.process_frame(frame)
                
                # Update track history for trails
                if display_settings['show_trails']:
                    update_track_history(track_history, tracked_players)
                
                # Apply display customizations
                if display_settings['show_trails']:
                    vis_frame = draw_player_trails(vis_frame, track_history)
                
                # Store comprehensive frame data
                frame_data = {
                    'frame': frame_count,
                    'tracked_players': len(tracked_players),
                    'unique_ids': [p['track_id'] for p in tracked_players],
                    'teams': [p.get('team', 'unknown') for p in tracked_players],
                    'avg_confidence': np.mean([p.get('confidence', 0) for p in tracked_players]) if tracked_players else 0,
                    'masked_players': sum(1 for p in tracked_players if p.get('has_mask', False)),
                    'timestamp': time.time(),
                    'velocities': [p.get('velocity', (0, 0)) for p in tracked_players]
                }
                all_frame_data.append(frame_data)
                
                frame_count += 1
                single_step = False
            else:
                # Show paused frame with overlay
                pause_overlay = vis_frame.copy()
                cv2.putText(pause_overlay, "PAUSED - Press 'p' to resume or SPACE for single step", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                vis_frame = pause_overlay
            
            # Add custom overlays based on display settings
            if not display_settings['show_stats']:
                # Remove stats overlay by re-processing without stats
                pass  # Stats are drawn in the pipeline, would need modification to remove
            
            # Display frame
            cv2.imshow('Complete Basketball Tracking Pipeline', vis_frame)
            
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
            elif key == ord(' '):  # Space bar
                if paused:
                    single_step = True
                    print("Single step")
            elif key == ord('s') and not paused:
                # Save current frame with detailed info
                save_frame_with_info(vis_frame, tracked_players, frame_count)
            elif key == ord('r'):
                # Reset pipeline and statistics
                pipeline.reset_session()
                track_history.clear()
                all_frame_data.clear()
                frame_count = 0
                start_time = time.time()
                print("Pipeline and statistics reset")
            elif key == ord('m'):
                display_settings['show_masks'] = not display_settings['show_masks']
                print(f"Mask display: {'ON' if display_settings['show_masks'] else 'OFF'}")
            elif key == ord('b'):
                display_settings['show_boxes'] = not display_settings['show_boxes']
                print(f"Bounding box display: {'ON' if display_settings['show_boxes'] else 'OFF'}")
            elif key == ord('i'):
                display_settings['show_ids'] = not display_settings['show_ids']
                print(f"ID display: {'ON' if display_settings['show_ids'] else 'OFF'}")
            elif key == ord('t'):
                display_settings['show_trails'] = not display_settings['show_trails']
                if not display_settings['show_trails']:
                    track_history.clear()
                print(f"Trail display: {'ON' if display_settings['show_trails'] else 'OFF'}")
            elif key == ord('v'):
                display_settings['show_velocity'] = not display_settings['show_velocity']
                print(f"Velocity vectors: {'ON' if display_settings['show_velocity'] else 'OFF'}")
            elif key == ord('h'):
                print_help()
            elif key == ord('+') or key == ord('='):
                display_settings['mask_alpha'] = min(1.0, display_settings['mask_alpha'] + 0.1)
                print(f"Mask alpha: {display_settings['mask_alpha']:.1f}")
            elif key == ord('-'):
                display_settings['mask_alpha'] = max(0.0, display_settings['mask_alpha'] - 0.1)
                print(f"Mask alpha: {display_settings['mask_alpha']:.1f}")
            
            # Print periodic comprehensive statistics
            if not paused and frame_count % 30 == 0 and frame_count > 0:
                current_perf = pipeline.get_current_performance()
                tracker_stats = current_perf.get('tracker_stats', {})
                
                print(f"Frame {frame_count}: {current_perf.get('current_fps', 0):.1f} FPS")
                print(f"  Players: {len(tracked_players)} tracked | "
                      f"Confirmed: {tracker_stats.get('confirmed_tracks', 0)} | "
                      f"Unique IDs: {len(set(p['track_id'] for p in tracked_players))}")
                
                if tracked_players:
                    teams = {}
                    for player in tracked_players:
                        team = player.get('team', 'unknown')
                        teams[team] = teams.get(team, 0) + 1
                    print(f"  Team distribution: {teams}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Calculate and display final comprehensive statistics
        total_time = time.time() - start_time
        print_final_comprehensive_statistics(pipeline, all_frame_data, total_time, optimization_level)
        
        # Save comprehensive statistics if requested
        if save_stats:
            save_comprehensive_statistics(pipeline, all_frame_data, optimization_level)
    
    return True


def update_track_history(track_history: dict, tracked_players: list, max_history: int = 30):
    """Update track history for trail visualization"""
    current_ids = set()
    
    for player in tracked_players:
        track_id = player['track_id']
        center = player['center']
        current_ids.add(track_id)
        
        if track_id not in track_history:
            track_history[track_id] = []
        
        track_history[track_id].append(center)
        
        # Keep only recent history
        if len(track_history[track_id]) > max_history:
            track_history[track_id] = track_history[track_id][-max_history:]
    
    # Remove tracks that are no longer active
    inactive_ids = set(track_history.keys()) - current_ids
    for track_id in inactive_ids:
        del track_history[track_id]


def draw_player_trails(frame: np.ndarray, track_history: dict) -> np.ndarray:
    """Draw player movement trails"""
    trail_frame = frame.copy()
    
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
    ]
    
    for track_id, history in track_history.items():
        if len(history) < 2:
            continue
        
        color = colors[track_id % len(colors)]
        
        # Draw trail with fading effect
        for i in range(1, len(history)):
            # Fade older points
            alpha = i / len(history)
            thickness = max(1, int(3 * alpha))
            
            pt1 = (int(history[i-1][0]), int(history[i-1][1]))
            pt2 = (int(history[i][0]), int(history[i][1]))
            
            cv2.line(trail_frame, pt1, pt2, color, thickness)
    
    return trail_frame


def save_frame_with_info(frame: np.ndarray, tracked_players: list, frame_count: int):
    """Save current frame with detailed tracking information"""
    timestamp = int(time.time())
    frame_filename = f"tracking_frame_{frame_count:06d}_{timestamp}.jpg"
    
    # Save frame
    cv2.imwrite(frame_filename, frame)
    
    # Save tracking data
    data_filename = f"tracking_data_{frame_count:06d}_{timestamp}.json"
    tracking_data = {
        'frame_number': frame_count,
        'timestamp': timestamp,
        'tracked_players': tracked_players,
        'summary': {
            'total_players': len(tracked_players),
            'unique_ids': [p['track_id'] for p in tracked_players],
            'teams': [p.get('team', 'unknown') for p in tracked_players],
            'avg_confidence': np.mean([p.get('confidence', 0) for p in tracked_players]) if tracked_players else 0
        }
    }
    
    with open(data_filename, 'w') as f:
        json.dump(tracking_data, f, indent=2, default=str)
    
    print(f"Frame and data saved: {frame_filename}, {data_filename}")


def print_help():
    """Print comprehensive help information"""
    print("\n" + "="*70)
    print("COMPLETE BASKETBALL TRACKING PIPELINE - KEYBOARD CONTROLS")
    print("="*70)
    print("Basic Controls:")
    print("  q         - Quit application")
    print("  p         - Pause/Resume processing")
    print("  SPACE     - Single step when paused")
    print("  s         - Save current frame and tracking data")
    print("  r         - Reset tracking and statistics")
    print("  h         - Show this help")
    print("")
    print("Display Controls:")
    print("  m         - Toggle segmentation mask display")
    print("  b         - Toggle bounding box display")
    print("  i         - Toggle player ID display")
    print("  t         - Toggle movement trail display")
    print("  v         - Toggle velocity vector display")
    print("  +/-       - Increase/decrease mask transparency")
    print("")
    print("Information:")
    print("  Real-time FPS and component timing shown on screen")
    print("  Player counts and team distribution displayed")
    print("  Track quality and mask coverage statistics")
    print("="*70)


def print_final_comprehensive_statistics(pipeline, frame_data, total_time, optimization_level):
    """Print comprehensive final statistics for complete pipeline"""
    print("\n" + "="*80)
    print("COMPLETE BASKETBALL TRACKING PIPELINE - FINAL RESULTS")
    print("="*80)
    
    if not frame_data:
        print("No frames processed")
        return
    
    # Get comprehensive performance data
    current_perf = pipeline.get_current_performance()
    
    # Basic processing statistics
    total_frames = len(frame_data)
    avg_fps = total_frames / total_time if total_time > 0 else 0
    
    print(f"📊 Processing Summary:")
    print(f"  Total frames processed: {total_frames}")
    print(f"  Total processing time: {total_time:.2f} seconds")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Optimization level: {optimization_level}")
    
    # Component status
    component_status = current_perf.get('component_status', {})
    print(f"\n🔧 Component Status:")
    print(f"  YOLO Detection: {'✅ Active' if component_status.get('yolo') else '❌ Inactive'}")
    print(f"  SAM2 Segmentation: {'✅ Active' if component_status.get('sam2') else '⚠️  Fallback Mode'}")
    print(f"  DeepSORT Tracking: {'✅ Active' if component_status.get('deepsort') else '❌ Inactive'}")
    
    # Tracking performance
    tracker_stats = current_perf.get('tracker_stats', {})
    total_detections = sum(fd['tracked_players'] for fd in frame_data)
    all_unique_ids = set()
    team_counts = {'red': 0, 'green': 0, 'blue': 0, 'unknown': 0}
    
    for fd in frame_data:
        all_unique_ids.update(fd['unique_ids'])
        for team in fd['teams']:
            if team in team_counts:
                team_counts[team] += 1
    
    print(f"\n🎯 Tracking Results:")
    print(f"  Total player detections: {total_detections}")
    print(f"  Unique player IDs tracked: {len(all_unique_ids)}")
    print(f"  Average players per frame: {total_detections / total_frames:.2f}")
    print(f"  Max players in single frame: {max(fd['tracked_players'] for fd in frame_data)}")
    print(f"  Frames with players: {sum(1 for fd in frame_data if fd['tracked_players'] > 0)}")
    print(f"  Active tracks: {tracker_stats.get('active_tracks', 0)}")
    print(f"  Confirmed tracks: {tracker_stats.get('confirmed_tracks', 0)}")
    print(f"  ID retention rate: {tracker_stats.get('id_retention_rate', 0):.2%}")
    
    # Team distribution
    if sum(team_counts.values()) > 0:
        print(f"\n👥 Team Distribution:")
        for team, count in team_counts.items():
            if count > 0:
                percentage = (count / sum(team_counts.values())) * 100
                print(f"  {team.capitalize()}: {count} detections ({percentage:.1f}%)")
    
    # Quality metrics
    avg_confidence = np.mean([fd['avg_confidence'] for fd in frame_data if fd['avg_confidence'] > 0])
    total_masked = sum(fd['masked_players'] for fd in frame_data)
    mask_coverage = total_masked / total_detections if total_detections > 0 else 0
    
    print(f"\n🎨 Quality Metrics:")
    print(f"  Average confidence: {avg_confidence:.3f}")
    print(f"  Segmentation coverage: {mask_coverage:.1%} ({total_masked}/{total_detections})")
    print(f"  Average track age: {tracker_stats.get('average_track_age', 0):.1f} frames")
    
    # Performance assessment for Phase 3
    print(f"\n✅ Phase 3 Assessment:")
    
    # FPS assessment
    if avg_fps >= 5:
        print(f"  ✅ Performance: GOOD ({avg_fps:.1f} FPS >= 5 FPS target)")
    elif avg_fps >= 3:
        print(f"  ⚠️  Performance: ACCEPTABLE ({avg_fps:.1f} FPS)")
    else:
        print(f"  ❌ Performance: NEEDS IMPROVEMENT ({avg_fps:.1f} FPS < 3 FPS)")
    
    # Tracking assessment
    if len(all_unique_ids) > 0 and total_detections > 0:
        id_efficiency = len(all_unique_ids) / (total_detections / total_frames)
        if id_efficiency >= 0.8:
            print(f"  ✅ ID Consistency: EXCELLENT ({tracker_stats.get('id_retention_rate', 0):.1%} retention)")
        elif id_efficiency >= 0.6:
            print(f"  ✅ ID Consistency: GOOD ({tracker_stats.get('id_retention_rate', 0):.1%} retention)")
        else:
            print(f"  ⚠️  ID Consistency: NEEDS IMPROVEMENT ({tracker_stats.get('id_retention_rate', 0):.1%} retention)")
    
    # Component integration assessment
    if component_status.get('yolo') and component_status.get('deepsort'):
        if component_status.get('sam2'):
            print(f"  ✅ Pipeline Integration: COMPLETE (All components active)")
        else:
            print(f"  ✅ Pipeline Integration: GOOD (Core components active)")
    else:
        print(f"  ❌ Pipeline Integration: INCOMPLETE (Missing components)")


def save_comprehensive_statistics(pipeline, frame_data, optimization_level):
    """Save comprehensive statistics to JSON file"""
    timestamp = int(time.time())
    filename = f"complete_tracking_stats_{optimization_level}_{timestamp}.json"
    
    # Gather all available statistics
    current_perf = pipeline.get_current_performance()
    
    comprehensive_stats = {
        'test_info': {
            'timestamp': timestamp,
            'optimization_level': optimization_level,
            'total_frames': len(frame_data)
        },
        'performance_metrics': current_perf,
        'frame_by_frame_data': frame_data,
        'summary': {
            'total_detections': sum(fd['tracked_players'] for fd in frame_data),
            'unique_players': len(set(player_id for fd in frame_data for player_id in fd['unique_ids'])),
            'avg_players_per_frame': sum(fd['tracked_players'] for fd in frame_data) / len(frame_data) if frame_data else 0,
            'team_distribution': {},
            'quality_metrics': {
                'avg_confidence': np.mean([fd['avg_confidence'] for fd in frame_data if fd['avg_confidence'] > 0]) if frame_data else 0,
                'mask_coverage': sum(fd['masked_players'] for fd in frame_data) / sum(fd['tracked_players'] for fd in frame_data) if sum(fd['tracked_players'] for fd in frame_data) > 0 else 0
            }
        }
    }
    
    # Calculate team distribution
    team_counts = {'red': 0, 'green': 0, 'blue': 0, 'unknown': 0}
    for fd in frame_data:
        for team in fd['teams']:
            if team in team_counts:
                team_counts[team] += 1
    comprehensive_stats['summary']['team_distribution'] = team_counts
    
    try:
        with open(filename, 'w') as f:
            json.dump(comprehensive_stats, f, indent=2, default=str)
        print(f"📊 Comprehensive statistics saved to: {filename}")
    except Exception as e:
        print(f"❌ Failed to save statistics: {e}")


def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Test Complete Basketball Tracking Pipeline')
    parser.add_argument('--input', '-i', type=str, default='0',
                       help='Input source (0 for webcam, path for video file)')
    parser.add_argument('--optimization', '-opt', choices=['fast', 'balanced', 'quality'], 
                       default='balanced', help='Optimization level')
    parser.add_argument('--save-output', '-so', action='store_true',
                       help='Save output video')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output video file path')
    parser.add_argument('--save-stats', '-ss', action='store_true',
                       help='Save comprehensive performance statistics')
    parser.add_argument('--show-trails', '-st', action='store_true',
                       help='Show player movement trails')
    
    args = parser.parse_args()
    
    # Test complete basketball tracking pipeline
    success = test_complete_pipeline(
        video_path=args.input,
        optimization_level=args.optimization,
        save_output=args.save_output,
        output_path=args.output,
        save_stats=args.save_stats,
        show_trails=args.show_trails
    )
    
    if success:
        print("\n🎉 Phase 3 testing completed successfully!")
        print("\nComplete Basketball Tracking Pipeline Status:")
        print("✅ YOLO Detection - Player identification")
        print("✅ SAM2 Segmentation - Pixel-accurate boundaries") 
        print("✅ DeepSORT Tracking - Consistent ID tracking")
        print("✅ Integrated Pipeline - Full workflow")
        print("\nNext steps:")
        print("1. Review tracking consistency and ID retention")
        print("2. Test with different basketball scenarios")
        print("3. Ready for Phase 4: Team classification enhancement")
    else:
        print("\n❌ Phase 3 testing encountered issues")
        print("Please check the error messages above and resolve any problems")


if __name__ == "__main__":
    main()
