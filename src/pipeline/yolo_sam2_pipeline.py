"""YOLO + SAM2 Pipeline for Basketball Player Tracking

This module combines YOLO detection with SAM2 segmentation for comprehensive
basketball player analysis with both bounding boxes and pixel-accurate masks.
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from src.detection.yolo_detector import YOLODetector
from src.segmentation.sam2_segmenter import SAM2Segmenter


class YOLOSAMPipeline:
    """Combined YOLO detection + SAM2 segmentation pipeline for basketball players"""
    
    def __init__(
        self, 
        yolo_config_path: Optional[str] = None,
        sam2_config_path: Optional[str] = None,
        optimization_level: str = "balanced"
    ):
        """
        Initialize the YOLO + SAM2 pipeline
        
        Args:
            yolo_config_path: Path to YOLO configuration file
            sam2_config_path: Path to SAM2 configuration file
            optimization_level: 'fast', 'balanced', or 'quality'
        """
        self.optimization_level = optimization_level
        
        # Initialize components
        print("🏀 Initializing YOLO + SAM2 Basketball Pipeline...")
        print(f"Optimization level: {optimization_level}")
        
        # Configure based on optimization level
        self._configure_optimization(optimization_level)
        
        # Initialize YOLO detector
        print("📦 Loading YOLO detector...")
        self.detector = YOLODetector(
            model_path=self.yolo_model_path,
            config_path=yolo_config_path or "config/yolo_config.yaml"
        )
        
        # Initialize SAM2 segmenter
        print("📦 Loading SAM2 segmenter...")
        try:
            self.segmenter = SAM2Segmenter(
                model_cfg=self.sam2_model_cfg,
                checkpoint=self.sam2_checkpoint,
                config_path=sam2_config_path or "config/sam2_config.yaml"
            )
            self.sam2_available = True
        except Exception as e:
            print(f"⚠️  SAM2 initialization failed: {e}")
            print("Falling back to bounding box mode...")
            self.segmenter = None
            self.sam2_available = False
        
        # Performance tracking
        self.frame_count = 0
        self.processing_times = {
            'detection': [],
            'segmentation': [],
            'total': []
        }
        
        # Visualization settings
        self.visualization_settings = {
            'show_boxes': True,
            'show_masks': True,
            'show_scores': True,
            'mask_alpha': 0.4,
            'box_thickness': 2
        }
        
        print("✅ Pipeline initialization complete!")
    
    def _configure_optimization(self, level: str):
        """Configure models based on optimization level"""
        if level == "fast":
            self.yolo_model_path = "yolo11n.pt"  # Nano - fastest
            self.sam2_model_cfg = "sam2_hiera_t.yaml"  # Tiny - fastest
            self.sam2_checkpoint = "sam2_hiera_tiny.pt"
            self.process_every_n_frames = 2  # Skip frames for speed
        elif level == "balanced":
            self.yolo_model_path = "yolo11n.pt"  # Nano
            self.sam2_model_cfg = "sam2_hiera_t.yaml"  # Tiny - good balance
            self.sam2_checkpoint = "sam2_hiera_tiny.pt"
            self.process_every_n_frames = 1  # Process all frames
        else:  # quality
            self.yolo_model_path = "yolo11s.pt"  # Small - better accuracy
            self.sam2_model_cfg = "sam2_hiera_l.yaml"  # Large
            self.sam2_checkpoint = "sam2_hiera_large.pt"
            self.process_every_n_frames = 1
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process single frame through YOLO + SAM2 pipeline
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (visualized_frame, segmented_detections)
        """
        start_time = time.time()
        
        # Check if we should process this frame (for optimization)
        if self.frame_count % self.process_every_n_frames != 0:
            self.frame_count += 1
            return frame, []
        
        try:
            # Step 1: YOLO Detection
            detection_start = time.time()
            detections = self.detector.detect_players(frame)
            detection_time = time.time() - detection_start
            self.processing_times['detection'].append(detection_time)
            
            # Step 2: SAM2 Segmentation
            segmentation_start = time.time()
            if detections and self.sam2_available and self.segmenter:
                segmented_detections = self.segmenter.segment_players(frame, detections)
            elif detections:
                # Fallback: create dummy masks
                segmented_detections = self._create_fallback_masks(frame, detections)
            else:
                segmented_detections = []
            
            segmentation_time = time.time() - segmentation_start
            self.processing_times['segmentation'].append(segmentation_time)
            
            # Step 3: Visualization
            vis_frame = self.visualize_results(frame, segmented_detections)
            
            # Update performance tracking
            total_time = time.time() - start_time
            self.processing_times['total'].append(total_time)
            self.frame_count += 1
            
            return vis_frame, segmented_detections
            
        except Exception as e:
            print(f"Warning: Frame processing failed: {e}")
            self.frame_count += 1
            return frame, []
    
    def _create_fallback_masks(self, frame: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """Create simple rectangular masks when SAM2 is not available"""
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
            segmented_detection['mask_type'] = 'bbox_fallback'
            
            segmented_detections.append(segmented_detection)
        
        return segmented_detections
    
    def visualize_results(
        self, 
        frame: np.ndarray, 
        detections: List[Dict],
        show_performance: bool = True
    ) -> np.ndarray:
        """
        Visualize detection and segmentation results
        
        Args:
            frame: Input frame
            detections: List of detections with masks
            show_performance: Whether to show performance info
            
        Returns:
            Visualized frame
        """
        vis_frame = frame.copy()
        
        if not detections:
            if show_performance:
                self._draw_performance_info(vis_frame)
            return vis_frame
        
        # Use SAM2 segmenter's visualization if available
        if self.segmenter:
            vis_frame = self.segmenter.visualize_segmentation(
                vis_frame,
                detections,
                alpha=self.visualization_settings['mask_alpha'],
                show_masks=self.visualization_settings['show_masks'],
                show_boxes=self.visualization_settings['show_boxes']
            )
        else:
            # Fallback visualization
            vis_frame = self._visualize_fallback(vis_frame, detections)
        
        # Add performance information
        if show_performance:
            self._draw_performance_info(vis_frame)
        
        # Add detection statistics
        self._draw_detection_stats(vis_frame, detections)
        
        return vis_frame
    
    def _visualize_fallback(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Fallback visualization when SAM2 is not available"""
        vis_frame = frame.copy()
        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            conf = detection['confidence']
            x1, y1, x2, y2 = bbox
            
            color = colors[i % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw mask if available
            if detection.get('has_mask', False) and 'mask' in detection:
                mask = detection['mask']
                colored_mask = np.zeros_like(vis_frame)
                colored_mask[mask] = color
                vis_frame = cv2.addWeighted(vis_frame, 0.7, colored_mask, 0.3, 0)
            
            # Draw label
            label = f"Player {i+1}: {conf:.2f}"
            cv2.putText(vis_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_frame
    
    def _draw_performance_info(self, frame: np.ndarray):
        """Draw performance information on frame"""
        if not self.processing_times['total']:
            return
        
        # Calculate recent performance
        recent_times = self.processing_times['total'][-30:]  # Last 30 frames
        avg_time = np.mean(recent_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        # Performance text
        perf_text = f"FPS: {fps:.1f} | Level: {self.optimization_level}"
        
        # SAM2 status
        sam2_status = "SAM2: ON" if self.sam2_available else "SAM2: OFF"
        perf_text += f" | {sam2_status}"
        
        # Draw performance info
        cv2.putText(frame, perf_text, (10, frame.shape[0] - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw detailed timing if available
        if (self.processing_times['detection'] and 
            self.processing_times['segmentation']):
            
            det_time = np.mean(self.processing_times['detection'][-10:]) * 1000
            seg_time = np.mean(self.processing_times['segmentation'][-10:]) * 1000
            
            timing_text = f"YOLO: {det_time:.1f}ms | SAM2: {seg_time:.1f}ms"
            cv2.putText(frame, timing_text, (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
    
    def _draw_detection_stats(self, frame: np.ndarray, detections: List[Dict]):
        """Draw detection statistics on frame"""
        # Count successful segmentations
        successful_masks = sum(1 for d in detections if d.get('has_mask', False))
        
        # Detection info
        stats_text = f"Players: {len(detections)} | Segmented: {successful_masks}"
        cv2.putText(frame, stats_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show mask types if mixed
        if detections:
            mask_types = set(d.get('mask_type', 'sam2') for d in detections if d.get('has_mask', False))
            if len(mask_types) > 1 or 'bbox_fallback' in mask_types:
                type_text = f"Masks: {', '.join(mask_types)}"
                cv2.putText(frame, type_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
    
    def process_video(
        self, 
        input_path: str, 
        output_path: Optional[str] = None,
        show_preview: bool = True
    ) -> Dict:
        """
        Process entire video through YOLO + SAM2 pipeline
        
        Args:
            input_path: Path to input video
            output_path: Path to save output video (optional)
            show_preview: Whether to show preview window
            
        Returns:
            Dictionary with processing statistics
        """
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {width}x{height} @ {fps} FPS ({total_frames} frames)")
        
        # Initialize video writer if output specified
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output will be saved to: {output_path}")
        
        # Processing statistics
        all_detections = []
        processing_start = time.time()
        
        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                vis_frame, detections = self.process_frame(frame)
                
                # Store detection data
                frame_data = {
                    'frame': frame_idx,
                    'detections': detections,
                    'player_count': len(detections),
                    'segmented_count': sum(1 for d in detections if d.get('has_mask', False))
                }
                all_detections.append(frame_data)
                
                # Write output frame
                if out:
                    out.write(vis_frame)
                
                # Show preview
                if show_preview:
                    cv2.imshow('YOLO + SAM2 Basketball Tracking', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Processing interrupted by user")
                        break
                
                # Progress update
                if frame_idx % 30 == 0:
                    progress = (frame_idx / total_frames) * 100
                    print(f"Progress: {progress:.1f}% - Frame {frame_idx}/{total_frames}")
                
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
        
        # Calculate final statistics
        processing_time = time.time() - processing_start
        
        stats = self._calculate_processing_statistics(all_detections, processing_time)
        
        print(f"\n✅ Video processing complete!")
        print(f"Processed {len(all_detections)} frames in {processing_time:.2f} seconds")
        print(f"Average FPS: {len(all_detections) / processing_time:.2f}")
        
        return stats
    
    def _calculate_processing_statistics(self, all_detections: List[Dict], processing_time: float) -> Dict:
        """Calculate comprehensive processing statistics"""
        if not all_detections:
            return {'error': 'No frames processed'}
        
        # Basic statistics
        total_frames = len(all_detections)
        total_players = sum(fd['player_count'] for fd in all_detections)
        total_segmented = sum(fd['segmented_count'] for fd in all_detections)
        
        # Performance statistics
        avg_fps = total_frames / processing_time if processing_time > 0 else 0
        
        performance_stats = {
            'detection': {},
            'segmentation': {},
            'total': {}
        }
        
        for component in performance_stats.keys():
            times = self.processing_times.get(component, [])
            if times:
                performance_stats[component] = {
                    'avg_time_ms': np.mean(times) * 1000,
                    'min_time_ms': np.min(times) * 1000,
                    'max_time_ms': np.max(times) * 1000,
                    'total_calls': len(times)
                }
        
        return {
            'processing_summary': {
                'total_frames': total_frames,
                'processing_time': processing_time,
                'average_fps': avg_fps,
                'optimization_level': self.optimization_level
            },
            'detection_summary': {
                'total_detections': total_players,
                'average_players_per_frame': total_players / total_frames,
                'max_players_in_frame': max(fd['player_count'] for fd in all_detections),
                'frames_with_players': sum(1 for fd in all_detections if fd['player_count'] > 0)
            },
            'segmentation_summary': {
                'total_segmentations': total_segmented,
                'segmentation_success_rate': total_segmented / total_players if total_players > 0 else 0,
                'sam2_available': self.sam2_available,
                'average_segmented_per_frame': total_segmented / total_frames
            },
            'performance_details': performance_stats,
            'frame_data': all_detections  # Full frame-by-frame data
        }
    
    def get_current_performance(self) -> Dict:
        """Get current performance metrics"""
        if not self.processing_times['total']:
            return {'status': 'No processing data available'}
        
        recent_times = self.processing_times['total'][-30:]
        avg_time = np.mean(recent_times)
        current_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'current_fps': current_fps,
            'frame_count': self.frame_count,
            'optimization_level': self.optimization_level,
            'sam2_available': self.sam2_available,
            'avg_detection_time_ms': np.mean(self.processing_times['detection'][-30:]) * 1000 
                                    if self.processing_times['detection'] else 0,
            'avg_segmentation_time_ms': np.mean(self.processing_times['segmentation'][-30:]) * 1000 
                                       if self.processing_times['segmentation'] else 0
        }
    
    def update_visualization_settings(self, **kwargs):
        """Update visualization settings"""
        self.visualization_settings.update(kwargs)
        print(f"Visualization settings updated: {kwargs}")
