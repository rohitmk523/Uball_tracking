"""Complete Basketball Player Tracking Pipeline

This module implements the complete basketball tracking system combining:
- YOLO11 for player detection
- SAM2 for pixel-accurate segmentation  
- DeepSORT for consistent ID tracking across frames

Optimized for basketball-specific scenarios with team classification.
"""

import cv2
import numpy as np
import time
import yaml
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json

from src.detection.yolo_detector import YOLODetector
from src.segmentation.sam2_segmenter import SAM2Segmenter
from src.tracking.deepsort_tracker import DeepSORTTracker

# ByteTrack is optional - only import if supervision is available
try:
    from src.tracking.bytetrack_tracker import ByteTrackTracker
    BYTETRACK_AVAILABLE = True
except ImportError:
    BYTETRACK_AVAILABLE = False


class BasketballTracker:
    """
    Complete basketball player tracking pipeline
    
    Integrates detection, segmentation, and tracking for comprehensive
    basketball player analysis with team classification and performance monitoring.
    """
    
    def __init__(
        self,
        yolo_config_path: Optional[str] = None,
        sam2_config_path: Optional[str] = None,
        deepsort_config_path: Optional[str] = None,
        optimization_level: str = "balanced"
    ):
        """
        Initialize complete basketball tracking pipeline
        
        Args:
            yolo_config_path: Path to YOLO configuration
            sam2_config_path: Path to SAM2 configuration  
            deepsort_config_path: Path to DeepSORT configuration
            optimization_level: 'fast', 'balanced', or 'quality'
        """
        self.optimization_level = optimization_level
        
        print("🏀 Initializing Complete Basketball Tracking Pipeline")
        print("=" * 60)
        print(f"Optimization level: {optimization_level}")
        
        # Configure optimization settings
        self._configure_optimization(optimization_level)
        
        # Initialize components
        self._initialize_components(yolo_config_path, sam2_config_path, deepsort_config_path)
        
        # Performance tracking
        self.frame_count = 0
        self.processing_times = {
            'detection': [],
            'segmentation': [],
            'tracking': [],
            'visualization': [],
            'total': []
        }
        
        # Team classification (will be enhanced in Phase 4)
        self.team_colors = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'unknown': (128, 128, 128)
        }
        
        # Statistics tracking
        self.session_stats = {
            'total_frames': 0,
            'total_detections': 0,
            'total_tracks': 0,
            'unique_players': set(),
            'team_stats': {'red': 0, 'green': 0, 'blue': 0, 'unknown': 0}
        }
        
        print("✅ Basketball tracking pipeline initialized successfully!")
        print(f"Components: YOLO ✅ | SAM2 {'✅' if self.sam2_available else '⚠️ '} | DeepSORT ✅")
    
    def _configure_optimization(self, level: str):
        """Configure pipeline based on optimization level"""
        if level == "fast":
            self.yolo_model = "yolo11s.pt"  # Better detection for tracking
            self.sam2_model = "sam2_hiera_t.yaml"
            self.sam2_checkpoint = "sam2_hiera_tiny.pt"
            self.process_every_n_frames = 2
            self.enable_segmentation = False
            self.enable_advanced_tracking = True
        
        elif level == "balanced":
            self.yolo_model = "yolo11m.pt"  # Small model - better detection
            self.sam2_model = "sam2_hiera_l.yaml"
            self.sam2_checkpoint = "sam2_hiera_large.pt"
            self.process_every_n_frames = 1
            self.enable_segmentation = True
            self.enable_advanced_tracking = True
        
        elif level == "quality":  # quality
            self.yolo_model = "yolo11l.pt"

            self.sam2_model = "sam2_hiera_l.yaml"
            self.sam2_checkpoint = "sam2_hiera_large.pt"
            self.process_every_n_frames = 1
            self.enable_segmentation = True
            self.enable_advanced_tracking = True
    
    def _initialize_components(self, yolo_config, sam2_config, deepsort_config):
        """Initialize all pipeline components"""
        
        # Initialize YOLO detector
        print("🎯 Initializing YOLO detector...")
        try:
            self.detector = YOLODetector(
                model_path=self.yolo_model,
                config_path=yolo_config or "config/yolo_config.yaml"
            )
            self.yolo_available = True
            print("✅ YOLO detector ready")
        except Exception as e:
            print(f"❌ YOLO initialization failed: {e}")
            raise RuntimeError("YOLO detector is required for the pipeline")
        
        # Initialize SAM2 segmenter
        print("🖼️  Initializing SAM2 segmenter...")
        try:
            if self.enable_segmentation:
                self.segmenter = SAM2Segmenter(
                    model_cfg=self.sam2_model,
                    checkpoint=self.sam2_checkpoint,
                    config_path=sam2_config or "config/sam2_config.yaml"
                )
                self.sam2_available = True
                print("✅ SAM2 segmenter ready")
            else:
                self.segmenter = None
                self.sam2_available = False
                print("⚠️  SAM2 disabled for fast mode")
        except Exception as e:
            print(f"⚠️  SAM2 initialization failed: {e}")
            print("Continuing with bounding box mode...")
            self.segmenter = None
            self.sam2_available = False
        
        # Initialize DeepSORT tracker
        print("👥 Initializing DeepSORT tracker...")
        try:
            self.tracker = DeepSORTTracker(
                config_path=deepsort_config or "config/deepsort_config.yaml"
            )
            self.tracking_available = True
            print("✅ DeepSORT tracker ready")
        except Exception as e:
            print(f"❌ DeepSORT initialization failed: {e}")
            raise RuntimeError("DeepSORT tracker is required for the pipeline")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process single frame through complete pipeline
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (visualized_frame, tracked_players)
        """
        start_time = time.time()
        
        # Check if we should process this frame
        if self.frame_count % self.process_every_n_frames != 0:
            self.frame_count += 1
            return frame, []
        
        try:
            # Step 1: YOLO Detection
            detection_start = time.time()
            detections = self.detector.detect_players(frame)
            detection_time = time.time() - detection_start
            self.processing_times['detection'].append(detection_time)
            
            # Step 2: SAM2 Segmentation (optional)
            segmentation_start = time.time()
            if detections and self.sam2_available and self.segmenter:
                segmented_detections = self.segmenter.segment_players(frame, detections)
            elif detections:
                # Create fallback masks
                segmented_detections = self._create_fallback_masks(frame, detections)
            else:
                segmented_detections = []
            
            segmentation_time = time.time() - segmentation_start
            self.processing_times['segmentation'].append(segmentation_time)
            
            # Step 3: DeepSORT Tracking
            tracking_start = time.time()
            if segmented_detections:
                tracked_players = self.tracker.update(segmented_detections, frame)
            else:
                tracked_players = []
            
            tracking_time = time.time() - tracking_start
            self.processing_times['tracking'].append(tracking_time)
            
            # Step 4: Enhance tracks with segmentation data
            enhanced_tracks = self._enhance_tracks_with_segmentation(
                tracked_players, segmented_detections
            )
            
            # Step 5: Visualization
            visualization_start = time.time()
            vis_frame = self.visualize_complete_tracking(frame, enhanced_tracks, segmented_detections)
            visualization_time = time.time() - visualization_start
            self.processing_times['visualization'].append(visualization_time)
            
            # Update statistics
            self._update_statistics(enhanced_tracks)
            
            # Record total processing time
            total_time = time.time() - start_time
            self.processing_times['total'].append(total_time)
            self.frame_count += 1
            
            return vis_frame, enhanced_tracks
            
        except Exception as e:
            print(f"Warning: Frame processing failed: {e}")
            self.frame_count += 1
            return frame, []
    
    def _create_fallback_masks(self, frame: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """Create rectangular masks when SAM2 is not available"""
        segmented_detections = []
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Create rectangular mask
            mask = np.zeros(frame.shape[:2], dtype=bool)
            mask[y1:y2, x1:x2] = True
            
            segmented_detection = detection.copy()
            segmented_detection['mask'] = mask
            segmented_detection['mask_score'] = 0.8
            segmented_detection['has_mask'] = True
            segmented_detection['mask_type'] = 'fallback'
            
            segmented_detections.append(segmented_detection)
        
        return segmented_detections
    
    def _enhance_tracks_with_segmentation(
        self, 
        tracked_players: List[Dict], 
        segmented_detections: List[Dict]
    ) -> List[Dict]:
        """Enhance tracking results with segmentation information"""
        
        # Create mapping from bbox to segmentation data
        bbox_to_segmentation = {}
        for seg_det in segmented_detections:
            bbox_key = tuple(seg_det['bbox'])
            bbox_to_segmentation[bbox_key] = seg_det
        
        enhanced_tracks = []
        for track in tracked_players:
            enhanced_track = track.copy()
            
            # Find matching segmentation
            track_bbox = track['bbox']
            bbox_key = tuple([int(x) for x in track_bbox])
            
            # Try to find exact match or closest match
            matching_seg = None
            min_distance = float('inf')
            
            for seg_det in segmented_detections:
                seg_bbox = seg_det['bbox']
                
                # Calculate center distance
                track_center = [(track_bbox[0] + track_bbox[2]) / 2, (track_bbox[1] + track_bbox[3]) / 2]
                seg_center = [(seg_bbox[0] + seg_bbox[2]) / 2, (seg_bbox[1] + seg_bbox[3]) / 2]
                
                distance = np.sqrt((track_center[0] - seg_center[0])**2 + 
                                 (track_center[1] - seg_center[1])**2)
                
                if distance < min_distance and distance < 50:  # 50 pixel threshold
                    min_distance = distance
                    matching_seg = seg_det
            
            # Add segmentation information if found
            if matching_seg:
                enhanced_track['mask'] = matching_seg.get('mask')
                enhanced_track['mask_score'] = matching_seg.get('mask_score', 0.0)
                enhanced_track['has_mask'] = matching_seg.get('has_mask', False)
                enhanced_track['mask_type'] = matching_seg.get('mask_type', 'sam2')
                enhanced_track['detection_confidence'] = matching_seg.get('confidence', 0.0)
            else:
                enhanced_track['has_mask'] = False
            
            enhanced_tracks.append(enhanced_track)
        
        return enhanced_tracks
    
    def _update_statistics(self, tracked_players: List[Dict]):
        """Update session statistics"""
        self.session_stats['total_frames'] += 1
        self.session_stats['total_detections'] += len(tracked_players)
        
        for player in tracked_players:
            track_id = player['track_id']
            self.session_stats['unique_players'].add(track_id)
            
            team = player.get('team', 'unknown')
            if team in self.session_stats['team_stats']:
                self.session_stats['team_stats'][team] += 1
    
    def visualize_complete_tracking(
        self, 
        frame: np.ndarray, 
        tracked_players: List[Dict],
        segmented_detections: List[Dict]
    ) -> np.ndarray:
        """
        Comprehensive visualization of tracking results
        
        Args:
            frame: Input frame
            tracked_players: List of tracked players with IDs
            segmented_detections: Original detections with masks
            
        Returns:
            Visualized frame with all information
        """
        vis_frame = frame.copy()
        
        if not tracked_players:
            self._draw_performance_info(vis_frame)
            return vis_frame
        
        # Draw each tracked player
        for player in tracked_players:
            track_id = player['track_id']
            bbox = player['bbox']
            team = player.get('team', 'unknown')
            confidence = player.get('confidence', 0.0)
            velocity = player.get('velocity', (0, 0))
            
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Get color based on team or track ID
            if team in self.team_colors:
                color = self.team_colors[team]
            else:
                # Use track ID for color consistency
                color_palette = [
                    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
                ]
                color = color_palette[track_id % len(color_palette)]
            
            # Draw segmentation mask if available
            if player.get('has_mask', False) and 'mask' in player:
                mask = player['mask']
                if mask is not None and np.sum(mask) > 0:
                    # Method 1: Direct pixel blending (more reliable)
                    alpha = 0.4  # Mask transparency
                    vis_frame[mask] = (vis_frame[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
                    
                    # Draw mask contour for better visibility
                    mask_uint8 = mask.astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis_frame, contours, -1, color, 2)
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw track information
            label_parts = [f"#{track_id}"]
            if team != 'unknown':
                label_parts.append(f"({team[0].upper()})")
            if confidence > 0:
                label_parts.append(f"{confidence:.2f}")
            
            label = " ".join(label_parts)
            
            # Draw label with background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0] + 5, y1), color, -1)
            cv2.putText(vis_frame, label, (x1 + 2, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw center point
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(vis_frame, (center_x, center_y), 4, color, -1)
            
            # Draw velocity vector if significant
            vx, vy = velocity
            if abs(vx) > 5 or abs(vy) > 5:  # Only draw if moving
                end_x = int(center_x + vx * 3)
                end_y = int(center_y + vy * 3)
                cv2.arrowedLine(vis_frame, (center_x, center_y), (end_x, end_y), 
                               color, 2, tipLength=0.3)
        
        # Draw comprehensive information overlay
        self._draw_tracking_info(vis_frame, tracked_players)
        self._draw_performance_info(vis_frame)
        
        return vis_frame
    
    def _draw_tracking_info(self, frame: np.ndarray, tracked_players: List[Dict]):
        """Draw tracking statistics and information"""
        # Count players by team
        team_counts = {'red': 0, 'green': 0, 'blue': 0, 'unknown': 0}
        for player in tracked_players:
            team = player.get('team', 'unknown')
            if team in team_counts:
                team_counts[team] += 1
        
        # Draw team statistics
        y_offset = 30
        total_players = len(tracked_players)
        
        info_text = f"Players: {total_players}"
        if team_counts['red'] > 0 or team_counts['green'] > 0 or team_counts['blue'] > 0:
            team_parts = []
            for team, count in team_counts.items():
                if count > 0 and team != 'unknown':
                    team_parts.append(f"{team[0].upper()}:{count}")
            if team_parts:
                info_text += f" ({', '.join(team_parts)})"
        
        cv2.putText(frame, info_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw tracking quality indicators
        if tracked_players:
            avg_confidence = np.mean([p.get('confidence', 0) for p in tracked_players])
            quality_text = f"Avg Conf: {avg_confidence:.2f}"
            
            # Add mask coverage info
            masked_players = sum(1 for p in tracked_players if p.get('has_mask', False))
            if masked_players > 0:
                quality_text += f" | Masks: {masked_players}/{total_players}"
            
            cv2.putText(frame, quality_text, (10, y_offset + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
    
    def _draw_performance_info(self, frame: np.ndarray):
        """Draw performance information"""
        if not self.processing_times['total']:
            return
        
        # Calculate recent performance (last 30 frames)
        recent_times = self.processing_times['total'][-30:]
        avg_time = np.mean(recent_times)
        current_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        # Component breakdown
        component_times = {}
        for component in ['detection', 'segmentation', 'tracking']:
            if self.processing_times[component]:
                component_times[component] = np.mean(self.processing_times[component][-10:]) * 1000
        
        # Main performance display
        perf_text = f"FPS: {current_fps:.1f} | {self.optimization_level.upper()}"
        cv2.putText(frame, perf_text, (10, frame.shape[0] - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Component timing
        if component_times:
            timing_parts = []
            for comp, time_ms in component_times.items():
                short_name = comp[:4].upper()  # DETE, SEGM, TRAC
                timing_parts.append(f"{short_name}:{time_ms:.0f}ms")
            
            timing_text = " | ".join(timing_parts)
            cv2.putText(frame, timing_text, (10, frame.shape[0] - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
        
        # Status indicators
        status_parts = []
        if self.sam2_available:
            status_parts.append("SAM2:ON")
        else:
            status_parts.append("SAM2:OFF")
        
        status_text = " | ".join(status_parts)
        cv2.putText(frame, status_text, (10, frame.shape[0] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    def process_video(
        self, 
        input_path: str, 
        output_path: Optional[str] = None,
        show_preview: bool = True,
        save_stats: bool = False
    ) -> Dict:
        """
        Process complete video through basketball tracking pipeline
        
        Args:
            input_path: Path to input video
            output_path: Path to save output video
            show_preview: Whether to show preview window
            save_stats: Whether to save detailed statistics
            
        Returns:
            Processing statistics and results
        """
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Processing video: {width}x{height} @ {fps} FPS ({total_frames} frames)")
        
        # Initialize video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 Output will be saved to: {output_path}")
        
        # Processing statistics
        all_frame_data = []
        processing_start = time.time()
        
        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                vis_frame, tracked_players = self.process_frame(frame)
                
                # Store frame data
                frame_data = {
                    'frame': frame_idx,
                    'tracked_players': len(tracked_players),
                    'unique_ids': [p['track_id'] for p in tracked_players],
                    'teams': [p.get('team', 'unknown') for p in tracked_players],
                    'avg_confidence': np.mean([p.get('confidence', 0) for p in tracked_players]) if tracked_players else 0,
                    'masked_players': sum(1 for p in tracked_players if p.get('has_mask', False))
                }
                all_frame_data.append(frame_data)
                
                # Write output
                if out:
                    out.write(vis_frame)
                
                # Show preview
                if show_preview:
                    cv2.imshow('Complete Basketball Tracking', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Processing interrupted by user")
                        break
                
                # Progress update
                if frame_idx % 30 == 0:
                    progress = (frame_idx / total_frames) * 100
                    print(f"Progress: {progress:.1f}% - Frame {frame_idx}/{total_frames} - Players: {len(tracked_players)}")
                
                frame_idx += 1
        
        except KeyboardInterrupt:
            print("Processing interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            if show_preview:
                cv2.destroyAllWindows()
        
        # Calculate comprehensive statistics
        processing_time = time.time() - processing_start
        stats = self._calculate_comprehensive_stats(all_frame_data, processing_time)
        
        # Save detailed statistics if requested
        if save_stats:
            stats_filename = f"basketball_tracking_stats_{int(time.time())}.json"
            with open(stats_filename, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            print(f"📊 Detailed statistics saved to: {stats_filename}")
        
        print(f"\n✅ Video processing complete!")
        print(f"Processed {len(all_frame_data)} frames in {processing_time:.2f} seconds")
        print(f"Average FPS: {len(all_frame_data) / processing_time:.2f}")
        
        return stats
    
    def _calculate_comprehensive_stats(self, frame_data: List[Dict], processing_time: float) -> Dict:
        """Calculate comprehensive processing statistics"""
        if not frame_data:
            return {'error': 'No frames processed'}
        
        # Basic statistics
        total_frames = len(frame_data)
        all_tracked_players = sum(fd['tracked_players'] for fd in frame_data)
        all_unique_ids = set()
        team_distribution = {'red': 0, 'green': 0, 'blue': 0, 'unknown': 0}
        
        for fd in frame_data:
            all_unique_ids.update(fd['unique_ids'])
            for team in fd['teams']:
                if team in team_distribution:
                    team_distribution[team] += 1
        
        # Performance statistics
        performance_stats = {}
        for component in self.processing_times:
            times = self.processing_times[component]
            if times:
                performance_stats[component] = {
                    'avg_time_ms': np.mean(times) * 1000,
                    'max_time_ms': np.max(times) * 1000,
                    'min_time_ms': np.min(times) * 1000,
                    'total_calls': len(times)
                }
        
        # Tracking quality metrics
        avg_confidence = np.mean([fd['avg_confidence'] for fd in frame_data if fd['avg_confidence'] > 0])
        mask_coverage = np.mean([fd['masked_players'] / max(1, fd['tracked_players']) for fd in frame_data])
        
        return {
            'processing_summary': {
                'total_frames': total_frames,
                'processing_time_seconds': processing_time,
                'average_fps': total_frames / processing_time,
                'optimization_level': self.optimization_level
            },
            'tracking_summary': {
                'total_player_detections': all_tracked_players,
                'unique_player_ids': len(all_unique_ids),
                'average_players_per_frame': all_tracked_players / total_frames,
                'max_players_in_frame': max(fd['tracked_players'] for fd in frame_data),
                'frames_with_players': sum(1 for fd in frame_data if fd['tracked_players'] > 0),
                'team_distribution': team_distribution
            },
            'quality_metrics': {
                'average_confidence': float(avg_confidence) if not np.isnan(avg_confidence) else 0.0,
                'mask_coverage_rate': float(mask_coverage),
                'sam2_enabled': self.sam2_available,
                'segmentation_success_rate': mask_coverage
            },
            'performance_details': performance_stats,
            'component_status': {
                'yolo': self.yolo_available,
                'sam2': self.sam2_available,
                'deepsort': self.tracking_available
            },
            'frame_by_frame_data': frame_data
        }
    
    def get_current_performance(self) -> Dict:
        """Get current performance metrics"""
        if not self.processing_times['total']:
            return {'status': 'No processing data available'}
        
        recent_times = self.processing_times['total'][-30:]
        current_fps = 1.0 / np.mean(recent_times) if recent_times else 0
        
        # Get tracker statistics
        tracker_stats = self.tracker.get_tracking_statistics() if self.tracking_available else {}
        
        return {
            'current_fps': current_fps,
            'frame_count': self.frame_count,
            'optimization_level': self.optimization_level,
            'component_status': {
                'yolo': self.yolo_available,
                'sam2': self.sam2_available,
                'deepsort': self.tracking_available
            },
            'tracker_stats': tracker_stats,
            'session_stats': {
                **self.session_stats,
                'unique_players': len(self.session_stats['unique_players'])
            }
        }
    
    def reset_session(self):
        """Reset all session data"""
        self.tracker.reset()
        self.frame_count = 0
        self.processing_times = {key: [] for key in self.processing_times}
        self.session_stats = {
            'total_frames': 0,
            'total_detections': 0,
            'total_tracks': 0,
            'unique_players': set(),
            'team_stats': {'red': 0, 'green': 0, 'blue': 0, 'unknown': 0}
        }
        print("Basketball tracking session reset")
    
    def export_session_data(self, filename: str):
        """Export complete session data"""
        session_data = {
            'performance': self.get_current_performance(),
            'tracker_data': self.tracker.get_tracking_statistics(),
            'processing_times': self.processing_times,
            'configuration': {
                'optimization_level': self.optimization_level,
                'yolo_model': self.yolo_model,
                'sam2_model': self.sam2_model if self.sam2_available else None,
                'components_enabled': {
                    'yolo': self.yolo_available,
                    'sam2': self.sam2_available,
                    'deepsort': self.tracking_available
                }
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        print(f"Session data exported to: {filename}")
