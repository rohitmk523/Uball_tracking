"""ByteTrack Tracker with SAM2 Integration

Uses pixel-perfect masks from SAM2 for more accurate tracking instead of just bounding boxes.
Combines ByteTrack's motion-based tracking with SAM2's precise segmentation.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import supervision as sv
from scipy.spatial.distance import cdist


class ByteTrackTracker:
    """Enhanced ByteTrack tracker that uses SAM2 masks for more accurate tracking"""
    
    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 50,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
        use_mask_features: bool = True,
        config_path: Optional[str] = None
    ):
        """
        Initialize Enhanced ByteTrack tracker
        
        Args:
            track_activation_threshold: Confidence threshold for track activation
            lost_track_buffer: Frames to keep lost tracks
            minimum_matching_threshold: IoU threshold for matching
            frame_rate: Video frame rate
            use_mask_features: Whether to use mask-based features for tracking
            config_path: Optional config file path
        """
        # Initialize base ByteTrack tracker
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate
        )
        
        # Enhanced tracking settings
        self.use_mask_features = use_mask_features
        self.mask_weight = 0.3  # Weight for mask-based matching
        self.bbox_weight = 0.7  # Weight for bbox-based matching
        
        # Basketball-specific settings
        self.min_box_area = 1000
        self.max_box_area = 50000
        self.min_aspect_ratio = 0.3
        self.max_aspect_ratio = 1.2
        
        # Tracking statistics
        self.frame_count = 0
        self.track_history = {}
        self.mask_features_cache = {}  # Cache mask features for tracks
        
        print("ByteTrack tracker initialized")
        print(f"Mask-based features: {'Enabled' if use_mask_features else 'Disabled'}")
        print(f"Activation threshold: {track_activation_threshold}")
        print(f"Lost track buffer: {lost_track_buffer}")
    
    def update(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        Update tracker with new detections using enhanced mask-based matching
        
        Args:
            detections: List of detections with masks from SAM2
            frame: Current frame
            
        Returns:
            List of tracked players with IDs
        """
        self.frame_count += 1
        
        if not detections:
            return []
        
        # Filter detections for basketball players
        filtered_detections = self._filter_detections(detections)
        
        if not filtered_detections:
            return []
        
        # Extract mask features if available
        if self.use_mask_features:
            self._extract_mask_features(filtered_detections)
        
        # Convert to supervision format
        sv_detections = self._convert_to_supervision(filtered_detections)
        
        # Get predictions from current tracks for enhanced matching
        if self.use_mask_features and hasattr(self.tracker, 'trackers'):
            self._enhance_tracking_with_masks(sv_detections, filtered_detections)
        
        # Update base tracker
        tracked_detections = self.tracker.update_with_detections(sv_detections)
        
        # Convert back to our format
        tracked_players = self._convert_from_supervision(
            tracked_detections, filtered_detections
        )
        
        # Update tracking history and mask features
        self._update_history(tracked_players, filtered_detections)
        
        return tracked_players
    
    def _extract_mask_features(self, detections: List[Dict]):
        """Extract features from SAM2 masks for enhanced tracking"""
        for detection in detections:
            if detection.get('has_mask', False) and 'mask' in detection:
                mask = detection['mask']
                if mask is not None and np.sum(mask) > 0:
                    # Extract mask-based features
                    features = self._compute_mask_features(mask, detection['bbox'])
                    detection['mask_features'] = features
    
    def _compute_mask_features(self, mask: np.ndarray, bbox: List[float]) -> Dict:
        """Compute features from mask for tracking"""
        # Mask centroid (more accurate than bbox center)
        mask_points = np.where(mask)
        if len(mask_points[0]) == 0:
            return {}
        
        centroid_y = np.mean(mask_points[0])
        centroid_x = np.mean(mask_points[1])
        
        # Mask area and shape features
        mask_area = np.sum(mask)
        
        # Mask bounding box (tighter than YOLO bbox)
        min_y, max_y = np.min(mask_points[0]), np.max(mask_points[0])
        min_x, max_x = np.min(mask_points[1]), np.max(mask_points[1])
        mask_bbox = [min_x, min_y, max_x, max_y]
        
        # Aspect ratio from mask
        mask_height = max_y - min_y
        mask_width = max_x - min_x
        mask_aspect_ratio = mask_width / mask_height if mask_height > 0 else 0
        
        # Shape compactness (how circular vs elongated)
        if mask_area > 0:
            perimeter = cv2.arcLength(
                cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0], 
                True
            ) if len(cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]) > 0 else 0
            compactness = 4 * np.pi * mask_area / (perimeter ** 2) if perimeter > 0 else 0
        else:
            compactness = 0
        
        return {
            'centroid': [centroid_x, centroid_y],
            'area': mask_area,
            'bbox': mask_bbox,
            'aspect_ratio': mask_aspect_ratio,
            'compactness': compactness,
            'shape_signature': self._compute_shape_signature(mask)
        }
    
    def _compute_shape_signature(self, mask: np.ndarray, num_points: int = 16) -> np.ndarray:
        """Compute a simple shape signature for mask matching"""
        # Find contour
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return np.zeros(num_points)
        
        # Get largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Sample points along contour
        if len(contour) < num_points:
            return np.zeros(num_points)
        
        # Resample contour to fixed number of points
        indices = np.linspace(0, len(contour) - 1, num_points, dtype=int)
        sampled_points = contour[indices].reshape(-1, 2)
        
        # Compute distances from centroid (simple shape signature)
        centroid = np.mean(sampled_points, axis=0)
        distances = np.linalg.norm(sampled_points - centroid, axis=1)
        
        # Normalize
        if np.max(distances) > 0:
            distances = distances / np.max(distances)
        
        return distances
    
    def _enhance_tracking_with_masks(self, sv_detections: sv.Detections, detections: List[Dict]):
        """Use mask features to improve tracking association"""
        # This is where we could implement custom association logic
        # For now, we'll store the mask features for post-processing
        pass
    
    def _calculate_mask_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity between two mask feature sets"""
        if not features1 or not features2:
            return 0.0
        
        # Centroid distance (normalized)
        centroid_dist = np.linalg.norm(
            np.array(features1['centroid']) - np.array(features2['centroid'])
        )
        centroid_sim = 1.0 / (1.0 + centroid_dist / 100.0)  # Normalize by typical distance
        
        # Area similarity
        area_ratio = min(features1['area'], features2['area']) / max(features1['area'], features2['area'])
        area_sim = area_ratio
        
        # Aspect ratio similarity
        aspect_diff = abs(features1['aspect_ratio'] - features2['aspect_ratio'])
        aspect_sim = 1.0 / (1.0 + aspect_diff)
        
        # Shape signature similarity
        shape_sim = 1.0 - np.mean(np.abs(features1['shape_signature'] - features2['shape_signature']))
        
        # Combined similarity
        similarity = (
            0.4 * centroid_sim +
            0.2 * area_sim +
            0.2 * aspect_sim +
            0.2 * shape_sim
        )
        
        return similarity
    
    def _filter_detections(self, detections: List[Dict]) -> List[Dict]:
        """Filter detections for valid basketball players"""
        filtered = []
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # Calculate box properties
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            aspect_ratio = width / height if height > 0 else 0
            
            # Filter by size and aspect ratio
            if (self.min_box_area <= area <= self.max_box_area and
                self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                filtered.append(detection)
        
        return filtered
    
    def _convert_to_supervision(self, detections: List[Dict]) -> sv.Detections:
        """Convert our detection format to supervision format"""
        if not detections:
            return sv.Detections.empty()
        
        # Use mask-based bounding boxes if available, otherwise use YOLO boxes
        xyxy = []
        for det in detections:
            if (self.use_mask_features and 
                det.get('mask_features') and 
                'bbox' in det['mask_features']):
                # Use tighter mask-based bounding box
                mask_bbox = det['mask_features']['bbox']
                xyxy.append(mask_bbox)
            else:
                # Use original YOLO bounding box
                xyxy.append(det['bbox'])
        
        xyxy = np.array(xyxy)
        
        # Extract confidences
        confidence = np.array([det['confidence'] for det in detections])
        
        # All detections are class 0 (person)
        class_id = np.zeros(len(detections), dtype=int)
        
        return sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )
    
    def _convert_from_supervision(
        self, 
        sv_detections: sv.Detections, 
        original_detections: List[Dict]
    ) -> List[Dict]:
        """Convert supervision format back to our format"""
        tracked_players = []
        
        if len(sv_detections) == 0:
            return tracked_players
        
        for i in range(len(sv_detections)):
            # Get original detection data
            original_idx = i if i < len(original_detections) else 0
            original_det = original_detections[original_idx]
            
            # Create tracked player
            tracked_player = {
                'track_id': int(sv_detections.tracker_id[i]),
                'bbox': sv_detections.xyxy[i].tolist(),
                'confidence': float(sv_detections.confidence[i]),
                'class_id': int(sv_detections.class_id[i]),
                'center': [
                    (sv_detections.xyxy[i][0] + sv_detections.xyxy[i][2]) / 2,
                    (sv_detections.xyxy[i][1] + sv_detections.xyxy[i][3]) / 2
                ],
                'area': (sv_detections.xyxy[i][2] - sv_detections.xyxy[i][0]) * 
                       (sv_detections.xyxy[i][3] - sv_detections.xyxy[i][1]),
                'frame_count': self.frame_count,
                'age': 1  # Will be updated in history
            }
            
            # Copy over additional data from original detection
            for key in ['mask', 'mask_score', 'has_mask', 'mask_type', 'mask_features']:
                if key in original_det:
                    tracked_player[key] = original_det[key]
            
            # Use mask centroid as center if available
            if (self.use_mask_features and 
                'mask_features' in tracked_player and 
                tracked_player['mask_features']):
                tracked_player['center'] = tracked_player['mask_features']['centroid']
            
            tracked_players.append(tracked_player)
        
        return tracked_players
    
    def _update_history(self, tracked_players: List[Dict], original_detections: List[Dict]):
        """Update tracking history and mask features cache"""
        current_ids = set()
        
        for player in tracked_players:
            track_id = player['track_id']
            current_ids.add(track_id)
            
            # Update basic history
            if track_id not in self.track_history:
                self.track_history[track_id] = {
                    'first_frame': self.frame_count,
                    'last_frame': self.frame_count,
                    'total_frames': 1,
                    'positions': [player['center']],
                    'mask_features_history': []
                }
            else:
                history = self.track_history[track_id]
                history['last_frame'] = self.frame_count
                history['total_frames'] += 1
                history['positions'].append(player['center'])
                
                # Keep only recent positions
                if len(history['positions']) > 30:
                    history['positions'] = history['positions'][-30:]
                
                player['age'] = history['total_frames']
            
            # Update mask features cache
            if self.use_mask_features and 'mask_features' in player:
                if track_id not in self.mask_features_cache:
                    self.mask_features_cache[track_id] = []
                
                self.mask_features_cache[track_id].append(player['mask_features'])
                
                # Keep only recent mask features
                if len(self.mask_features_cache[track_id]) > 10:
                    self.mask_features_cache[track_id] = self.mask_features_cache[track_id][-10:]
        
        # Clean up old tracks
        if self.frame_count % 100 == 0:
            old_ids = []
            for track_id, history in self.track_history.items():
                if self.frame_count - history['last_frame'] > 100:
                    old_ids.append(track_id)
            
            for track_id in old_ids:
                if track_id in self.track_history:
                    del self.track_history[track_id]
                if track_id in self.mask_features_cache:
                    del self.mask_features_cache[track_id]
    
    def get_track_statistics(self) -> Dict:
        """Get enhanced tracking statistics"""
        active_tracks = len([
            track_id for track_id, history in self.track_history.items()
            if self.frame_count - history['last_frame'] < 10
        ])
        
        return {
            'frame_count': self.frame_count,
            'total_tracks': len(self.track_history),
            'active_tracks': active_tracks,
            'mask_features_enabled': self.use_mask_features,
            'cached_features': len(self.mask_features_cache),
            'track_history': dict(self.track_history)
        }
