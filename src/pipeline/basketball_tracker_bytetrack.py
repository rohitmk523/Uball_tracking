"""Basketball Tracker with ByteTrack

A simpler, faster alternative to DeepSORT-based tracking.
Uses ByteTrack for motion-based tracking without appearance features.
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
from src.tracking.bytetrack_tracker import ByteTrackTracker


class BasketballTrackerByteTrack:
    """
    Basketball player tracking pipeline using ByteTrack
    
    Simpler and faster than DeepSORT-based approach.
    """
    
    def __init__(
        self,
        yolo_config_path: Optional[str] = None,
        sam2_config_path: Optional[str] = None,
        optimization_level: str = "balanced",
        mask_only_visualization: bool = False
    ):
        """
        Initialize ByteTrack-based basketball tracking pipeline
        
        Args:
            yolo_config_path: Path to YOLO configuration
            sam2_config_path: Path to SAM2 configuration  
            optimization_level: 'fast', 'balanced', or 'quality'
            mask_only_visualization: If True, show only segmentation masks without bounding boxes
        """
        self.optimization_level = optimization_level
        self.mask_only_visualization = mask_only_visualization
        
        print("🏀 Initializing ByteTrack Basketball Tracking Pipeline")
        print("=" * 60)
        print(f"Optimization level: {optimization_level}")
        print(f"Mask-only visualization: {mask_only_visualization}")
        
        # Configure optimization settings
        self._configure_optimization(optimization_level)
        
        # Initialize components
        self._initialize_components(yolo_config_path, sam2_config_path)
        
        # Performance tracking
        self.frame_count = 0
        self.processing_times = {
            'detection': [],
            'segmentation': [],
            'tracking': [],
            'visualization': [],
            'total': []
        }
        
        # Team classification (simple color-based)
        self.team_colors = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'cyan': (255, 255, 0),
            'magenta': (255, 0, 255),
            'yellow': (0, 255, 255),
            'purple': (128, 0, 128),
            'orange': (0, 165, 255),
            'unknown': (128, 128, 128)
        }
        
        # Statistics tracking
        self.session_stats = {
            'total_frames': 0,
            'total_detections': 0,
            'unique_tracks': set(),
            'max_simultaneous_players': 0
        }
        
        print("✅ ByteTrack basketball tracking pipeline initialized successfully!")
        print(f"Components: YOLO ✅ | SAM2 {'✅' if self.sam2_available else '⚠️ '} | ByteTrack ✅")
    
    def _configure_optimization(self, level: str):
        """Configure pipeline based on optimization level"""
        if level == "fast":
            self.yolo_model = "yolo11s.pt"  # Small model for better detection
            self.sam2_model = "sam2_hiera_t.yaml"
            self.sam2_checkpoint = "sam2_hiera_tiny.pt"
            self.process_every_n_frames = 1
            self.enable_segmentation = False  # YOLO-only mode
            self.enable_advanced_tracking = True
            # ByteTrack settings for speed
            self.track_activation_threshold = 0.2
            self.lost_track_buffer = 30
            self.minimum_matching_threshold = 0.7
        
        elif level == "balanced":
            self.yolo_model = "yolo11m.pt"
            self.sam2_model = "sam2_hiera_t.yaml"
            self.sam2_checkpoint = "sam2_hiera_tiny.pt"
            self.process_every_n_frames = 1
            self.enable_segmentation = True
            self.enable_advanced_tracking = True
            # ByteTrack settings for balance
            self.track_activation_threshold = 0.25
            self.lost_track_buffer = 50
            self.minimum_matching_threshold = 0.8
        
        elif level == "quality":  # quality
            self.yolo_model = "yolo11l.pt"
            self.sam2_model = "sam2_hiera_l.yaml"
            self.sam2_checkpoint = "sam2_hiera_large.pt"
            self.process_every_n_frames = 1
            self.enable_segmentation = True
            self.enable_advanced_tracking = True
            # ByteTrack settings for quality
            self.track_activation_threshold = 0.3
            self.lost_track_buffer = 60
            self.minimum_matching_threshold = 0.85
    
    def _initialize_components(self, yolo_config, sam2_config):
        """Initialize all pipeline components"""
        
        # Initialize YOLO detector
        print("🎯 Initializing YOLO detector...")
        try:
            # Construct full path to model
            model_path = f"models/yolo/{self.yolo_model}"
            self.detector = YOLODetector(
                model_path=model_path,
                config_path=yolo_config or "config/yolo_config.yaml"
            )
            self.yolo_available = True
            print("✅ YOLO detector ready")
        except Exception as e:
            print(f"❌ YOLO initialization failed: {e}")
            raise
        
        # Initialize SAM2 segmenter (optional)
        if self.enable_segmentation:
            print("🖼️  Initializing SAM2 segmenter...")
            try:
                self.segmenter = SAM2Segmenter(
                    model_cfg=self.sam2_model,
                    checkpoint=self.sam2_checkpoint,
                    config_path=sam2_config or "config/sam2_config.yaml"
                )
                self.sam2_available = True
                print("✅ SAM2 segmenter ready")
            except Exception as e:
                print(f"⚠️  SAM2 initialization failed: {e}")
                print("Continuing with bounding box mode...")
                self.segmenter = None
                self.sam2_available = False
        else:
            self.segmenter = None
            self.sam2_available = False
            print("⚠️  SAM2 segmentation disabled for speed")
        
        # Initialize ByteTrack tracker
        print("👥 Initializing ByteTrack tracker...")
        try:
            self.tracker = ByteTrackTracker(
                track_activation_threshold=self.track_activation_threshold,
                lost_track_buffer=self.lost_track_buffer,
                minimum_matching_threshold=self.minimum_matching_threshold,
                frame_rate=30
            )
            self.tracker_available = True
            print("✅ ByteTrack tracker ready")
        except Exception as e:
            print(f"❌ ByteTrack initialization failed: {e}")
            raise
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process single frame through ByteTrack pipeline
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (visualized_frame, tracked_players)
        """
        start_time = time.time()
        
        try:
            # Step 1: YOLO Detection
            detection_start = time.time()
            detections = self.detector.detect_players(frame)
            detection_time = time.time() - detection_start
            self.processing_times['detection'].append(detection_time)
            
            # Step 2: SAM2 Segmentation (optional)
            segmentation_start = time.time()
            if self.sam2_available and self.segmenter and detections:
                segmented_detections = self.segmenter.segment_players(frame, detections)
            else:
                # Create fallback masks for visualization
                segmented_detections = self._create_fallback_masks(frame, detections)
            
            segmentation_time = time.time() - segmentation_start
            self.processing_times['segmentation'].append(segmentation_time)
            
            # Step 3: ByteTrack Tracking
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
            if self.mask_only_visualization:
                vis_frame = self.visualize_masks_only(frame, enhanced_tracks)
            else:
                vis_frame = self.visualize_tracking(frame, enhanced_tracks)
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
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Create rectangular mask
            mask = np.zeros(frame.shape[:2], dtype=bool)
            mask[y1:y2, x1:x2] = True
            
            # Add mask information
            detection_with_mask = detection.copy()
            detection_with_mask.update({
                'mask': mask,
                'mask_score': 0.8,
                'has_mask': True,
                'mask_type': 'fallback'
            })
            
            segmented_detections.append(detection_with_mask)
        
        return segmented_detections
    
    def _enhance_tracks_with_segmentation(
        self, 
        tracked_players: List[Dict], 
        segmented_detections: List[Dict]
    ) -> List[Dict]:
        """Enhance tracking results with segmentation information"""
        
        enhanced_tracks = []
        for track in tracked_players:
            enhanced_track = track.copy()
            
            # Find matching segmentation by bbox proximity
            track_bbox = track['bbox']
            track_center = [(track_bbox[0] + track_bbox[2]) / 2, (track_bbox[1] + track_bbox[3]) / 2]
            
            matching_seg = None
            min_distance = float('inf')
            
            for seg_det in segmented_detections:
                seg_bbox = seg_det['bbox']
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
    
    def visualize_masks_only(self, frame: np.ndarray, tracked_players: List[Dict]) -> np.ndarray:
        """Visualize only segmentation masks without bounding boxes"""
        vis_frame = frame.copy()
        
        if not tracked_players:
            self._draw_performance_info(vis_frame)
            return vis_frame
        
        # Color palette for different track IDs
        color_palette = list(self.team_colors.values())[:-1]  # Exclude 'unknown'
        
        # Draw each tracked player - MASKS ONLY
        for player in tracked_players:
            track_id = player['track_id']
            
            # Get consistent color for track ID
            color = color_palette[track_id % len(color_palette)]
            
            # Draw segmentation mask if available
            if player.get('has_mask', False) and 'mask' in player:
                mask = player['mask']
                if mask is not None and np.sum(mask) > 0:
                    # Create a more vibrant mask overlay
                    alpha = 0.6  # Slightly more opaque for better visibility
                    vis_frame[mask] = (vis_frame[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
                    
                    # Draw mask contour for better definition
                    mask_uint8 = mask.astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis_frame, contours, -1, color, 3)
                    
                    # Optional: Add a small track ID label at the center of the mask
                    if len(contours) > 0:
                        # Find the centroid of the largest contour
                        largest_contour = max(contours, key=cv2.contourArea)
                        M = cv2.moments(largest_contour)
                        if M["m00"] != 0:
                            center_x = int(M["m10"] / M["m00"])
                            center_y = int(M["m01"] / M["m00"])
                            
                            # Draw a small circle with track ID
                            cv2.circle(vis_frame, (center_x, center_y), 15, color, -1)
                            cv2.putText(vis_frame, str(track_id), (center_x - 8, center_y + 5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                # Fallback: if no mask available, show a minimal indicator
                bbox = player['bbox']
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Draw a small circle to indicate tracked player without mask
                cv2.circle(vis_frame, (center_x, center_y), 8, color, -1)
                cv2.putText(vis_frame, str(track_id), (center_x - 5, center_y + 3), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw performance info
        self._draw_performance_info(vis_frame)
        
        return vis_frame
    
    def visualize_tracking(self, frame: np.ndarray, tracked_players: List[Dict]) -> np.ndarray:
        """Visualize ByteTrack tracking results"""
        vis_frame = frame.copy()
        
        if not tracked_players:
            self._draw_performance_info(vis_frame)
            return vis_frame
        
        # Color palette for different track IDs
        color_palette = list(self.team_colors.values())[:-1]  # Exclude 'unknown'
        
        # Draw each tracked player
        for player in tracked_players:
            track_id = player['track_id']
            bbox = player['bbox']
            confidence = player.get('confidence', 0.0)
            
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Get consistent color for track ID
            color = color_palette[track_id % len(color_palette)]
            
            # Draw segmentation mask if available
            if player.get('has_mask', False) and 'mask' in player:
                mask = player['mask']
                if mask is not None and np.sum(mask) > 0:
                    # Direct pixel blending for colored masks
                    alpha = 0.4
                    vis_frame[mask] = (vis_frame[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
                    
                    # Draw mask contour
                    mask_uint8 = mask.astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis_frame, contours, -1, color, 2)
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw track information
            label = f"Player {track_id} ({confidence:.2f})"
            
            # Draw label with background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0] + 5, y1), color, -1)
            cv2.putText(vis_frame, label, (x1 + 2, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw center point
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(vis_frame, (center_x, center_y), 4, color, -1)
        
        # Draw performance info
        self._draw_performance_info(vis_frame)
        
        return vis_frame
    
    def _draw_performance_info(self, frame: np.ndarray):
        """Draw performance information on frame"""
        if not self.processing_times['total']:
            return
        
        # Calculate recent performance
        recent_times = self.processing_times['total'][-10:]
        avg_time = np.mean(recent_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        # Performance text
        perf_text = f"FPS: {fps:.1f} | ByteTrack | Frame: {self.frame_count}"
        
        # Draw background
        text_size = cv2.getTextSize(perf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (10, 10), (text_size[0] + 20, text_size[1] + 20), (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, perf_text, (15, text_size[1] + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def _update_statistics(self, tracked_players: List[Dict]):
        """Update session statistics"""
        self.session_stats['total_frames'] += 1
        self.session_stats['total_detections'] += len(tracked_players)
        
        for player in tracked_players:
            track_id = player['track_id']
            self.session_stats['unique_tracks'].add(track_id)
        
        current_players = len(tracked_players)
        self.session_stats['max_simultaneous_players'] = max(
            self.session_stats['max_simultaneous_players'], 
            current_players
        )
    
    def get_statistics(self) -> Dict:
        """Get comprehensive tracking statistics"""
        if not self.processing_times['total']:
            return {}
        
        # Calculate performance metrics
        avg_detection_time = np.mean(self.processing_times['detection'][-100:])
        avg_segmentation_time = np.mean(self.processing_times['segmentation'][-100:])
        avg_tracking_time = np.mean(self.processing_times['tracking'][-100:])
        avg_total_time = np.mean(self.processing_times['total'][-100:])
        avg_fps = 1.0 / avg_total_time if avg_total_time > 0 else 0
        
        # Get ByteTrack statistics
        tracker_stats = self.tracker.get_track_statistics()
        
        return {
            'method': 'ByteTrack',
            'optimization_level': self.optimization_level,
            'performance': {
                'avg_fps': avg_fps,
                'avg_detection_time_ms': avg_detection_time * 1000,
                'avg_segmentation_time_ms': avg_segmentation_time * 1000,
                'avg_tracking_time_ms': avg_tracking_time * 1000,
                'avg_total_time_ms': avg_total_time * 1000
            },
            'tracking': {
                'total_frames': self.session_stats['total_frames'],
                'total_detections': self.session_stats['total_detections'],
                'unique_tracks': len(self.session_stats['unique_tracks']),
                'max_simultaneous_players': self.session_stats['max_simultaneous_players'],
                'active_tracks': tracker_stats.get('active_tracks', 0)
            },
            'components': {
                'yolo_model': self.yolo_model,
                'sam2_enabled': self.sam2_available,
                'tracker_type': 'ByteTrack'
            }
        }
