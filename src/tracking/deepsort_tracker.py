"""DeepSORT Tracker for Basketball Players

This module implements the DeepSORT tracking algorithm optimized for basketball
players, combining Kalman filtering for motion prediction with deep appearance
features for robust re-identification across frames.
"""

import numpy as np
import cv2
import yaml
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict, deque
import time

from .kalman_filter import BasketballKalmanFilter, KalmanTrackManager
from .feature_extractor import PlayerAppearanceExtractor, SimpleFeatureExtractor
from .utils import (
    hungarian_assignment, 
    compute_motion_distance_matrix,
    compute_appearance_distance_matrix,
    combine_distance_matrices,
    validate_track_detection_pair,
    non_maximum_suppression_tracks
)


class Track:
    """Individual track for a basketball player"""
    
    def __init__(self, track_id: int, detection: Dict, feature: np.ndarray):
        """
        Initialize track
        
        Args:
            track_id: Unique track identifier
            detection: Initial detection dictionary
            feature: Initial appearance feature
        """
        self.track_id = track_id
        self.detection = detection
        
        # Initialize Kalman filter
        self.kf = BasketballKalmanFilter(detection['bbox'])
        
        # Appearance features
        self.features = deque([feature], maxlen=10)  # Keep last 10 features
        self.current_feature = feature
        
        # Track state
        self.state = 'tentative'  # tentative, confirmed, deleted
        self.age = 0
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0
        
        # Track history
        self.history = deque(maxlen=30)
        self.history.append(detection['bbox'])
        
        # Basketball-specific attributes
        self.team = detection.get('team', 'unknown')
        self.confidence_history = deque([detection['confidence']], maxlen=10)
        
        # Performance tracking
        self.creation_time = time.time()
        self.last_update_time = time.time()
    
    def predict(self):
        """Predict next state"""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
    
    def update(self, detection: Dict, feature: np.ndarray):
        """Update track with new detection"""
        self.kf.update(detection['bbox'])
        
        # Update features
        self.features.append(feature)
        self.current_feature = feature
        
        # Update state
        self.detection = detection
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        self.last_update_time = time.time()
        
        # Update history
        self.history.append(detection['bbox'])
        self.confidence_history.append(detection['confidence'])
        
        # Update team if provided
        if 'team' in detection:
            self.team = detection['team']
        
        # Confirm track if enough hits
        if self.state == 'tentative' and self.hits >= 3:
            self.state = 'confirmed'
    
    def mark_missed(self):
        """Mark track as missed this frame"""
        self.hit_streak = 0
        
        # Delete track if too many misses
        if self.time_since_update > 10:
            self.state = 'deleted'
    
    def get_state(self) -> Dict:
        """Get current track state"""
        bbox = self.kf.get_state()
        velocity = self.kf.get_velocity()
        
        return {
            'track_id': self.track_id,
            'bbox': bbox,
            'velocity': velocity,
            'confidence': np.mean(list(self.confidence_history)),
            'age': self.age,
            'hits': self.hits,
            'hit_streak': self.hit_streak,
            'time_since_update': self.time_since_update,
            'state': self.state,
            'team': self.team,
            'feature': self.current_feature.copy(),
            'center': [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
            'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        }
    
    def get_average_feature(self) -> np.ndarray:
        """Get average appearance feature"""
        if not self.features:
            return np.zeros(256, dtype=np.float32)
        
        features_array = np.stack(list(self.features))
        avg_feature = np.mean(features_array, axis=0)
        
        # L2 normalize
        norm = np.linalg.norm(avg_feature)
        if norm > 0:
            avg_feature = avg_feature / norm
        
        return avg_feature
    
    def is_confirmed(self) -> bool:
        """Check if track is confirmed"""
        return self.state == 'confirmed'
    
    def is_deleted(self) -> bool:
        """Check if track is deleted"""
        return self.state == 'deleted'


class DeepSORTTracker:
    """
    DeepSORT tracker optimized for basketball players
    
    Combines Kalman filtering for motion prediction with deep appearance features
    for robust tracking across occlusions and appearance changes.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize DeepSORT tracker
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        if config_path:
            self.config = self._load_config(config_path)
        else:
            self.config = self._default_config()
        
        # Tracker parameters
        self.max_age = self.config.get('tracker', {}).get('max_age', 30)
        self.min_hits = self.config.get('tracker', {}).get('min_hits', 3)
        self.max_iou_distance = self.config.get('tracker', {}).get('max_iou_distance', 0.7)
        self.max_cosine_distance = self.config.get('tracker', {}).get('max_cosine_distance', 0.5)
        
        # Association parameters
        association_config = self.config.get('association', {})
        self.appearance_weight = association_config.get('appearance_weight', 0.7)
        self.motion_weight = association_config.get('motion_weight', 0.3)
        self.assignment_algorithm = association_config.get('algorithm', 'hungarian')
        
        # Initialize feature extractor
        self._initialize_feature_extractor()
        
        # Track management
        self.tracks = []
        self.next_id = 1
        self.deleted_tracks = []
        
        # Performance tracking
        self.frame_count = 0
        self.processing_times = {
            'feature_extraction': [],
            'prediction': [],
            'association': [],
            'update': []
        }
        
        # Basketball-specific parameters
        self.max_players = 15  # Maximum players on court
        self.team_consistency = self.config.get('basketball', {}).get('team_consistency', True)
        
        print(f"DeepSORT tracker initialized")
        print(f"Max age: {self.max_age}, Min hits: {self.min_hits}")
        print(f"Feature extractor: {'CNN' if hasattr(self.feature_extractor, 'model') else 'Simple'}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found. Using default config.")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'tracker': {
                'max_age': 30,
                'min_hits': 3,
                'max_iou_distance': 0.7,
                'max_cosine_distance': 0.5
            },
            'association': {
                'algorithm': 'hungarian',
                'appearance_weight': 0.7,
                'motion_weight': 0.3
            },
            'feature_extractor': {
                'model_type': 'simple_cnn',
                'feature_dim': 256,
                'input_size': [128, 64]
            },
            'basketball': {
                'team_consistency': True,
                'occlusion_handling': True
            }
        }
    
    def _initialize_feature_extractor(self):
        """Initialize feature extractor"""
        feature_config = self.config.get('feature_extractor', {})
        model_type = feature_config.get('model_type', 'simple_cnn')
        feature_dim = feature_config.get('feature_dim', 256)
        input_size = tuple(feature_config.get('input_size', [128, 64]))
        
        try:
            if model_type in ['cnn', 'simple_cnn']:
                self.feature_extractor = PlayerAppearanceExtractor(
                    feature_dim=feature_dim,
                    input_size=input_size
                )
                print("✅ CNN feature extractor initialized")
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        
        except Exception as e:
            print(f"⚠️  CNN feature extractor failed: {e}")
            print("Falling back to simple feature extractor")
            self.feature_extractor = SimpleFeatureExtractor(feature_dim=feature_dim)
    
    def update(self, detections: List[Dict], image: np.ndarray) -> List[Dict]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detections with 'bbox', 'confidence', etc.
            image: Current frame image
            
        Returns:
            List of confirmed tracks
        """
        start_time = time.time()
        
        # Step 1: Predict all existing tracks
        prediction_start = time.time()
        for track in self.tracks:
            track.predict()
        self.processing_times['prediction'].append(time.time() - prediction_start)
        
        # Step 2: Extract features for detections
        feature_start = time.time()
        if detections:
            bboxes = [det['bbox'] for det in detections]
            masks = [det.get('mask') for det in detections]
            detection_features = self.feature_extractor.extract_features(image, bboxes, masks)
        else:
            detection_features = []
        self.processing_times['feature_extraction'].append(time.time() - feature_start)
        
        # Step 3: Data association
        association_start = time.time()
        matches, unmatched_tracks, unmatched_detections = self._associate_detections_to_tracks(
            detections, detection_features
        )
        self.processing_times['association'].append(time.time() - association_start)
        
        # Step 4: Update matched tracks
        update_start = time.time()
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx], detection_features[det_idx])
        
        # Step 5: Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # Step 6: Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            if len(self.tracks) < self.max_players:
                new_track = Track(self.next_id, detections[det_idx], detection_features[det_idx])
                self.tracks.append(new_track)
                self.next_id += 1
        
        # Step 7: Remove deleted tracks
        active_tracks = []
        for track in self.tracks:
            if track.is_deleted():
                self.deleted_tracks.append(track)
            else:
                active_tracks.append(track)
        self.tracks = active_tracks
        
        self.processing_times['update'].append(time.time() - update_start)
        
        # Step 8: Get confirmed tracks for output
        confirmed_tracks = [track.get_state() for track in self.tracks if track.is_confirmed()]
        
        self.frame_count += 1
        
        return confirmed_tracks
    
    def _associate_detections_to_tracks(
        self, 
        detections: List[Dict], 
        detection_features: List[np.ndarray]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections to tracks using motion and appearance cues
        
        Args:
            detections: List of detections
            detection_features: List of detection features
            
        Returns:
            Tuple of (matches, unmatched_tracks, unmatched_detections)
        """
        if not self.tracks or not detections:
            return [], list(range(len(self.tracks))), list(range(len(detections)))
        
        # Get track predictions and features
        track_predictions = []
        track_features = []
        
        for track in self.tracks:
            track_state = track.get_state()
            track_predictions.append(track_state)
            track_features.append(track.get_average_feature())
        
        # Compute distance matrices
        motion_distances = compute_motion_distance_matrix(
            track_predictions, detections, distance_type="combined"
        )
        
        appearance_distances = compute_appearance_distance_matrix(
            track_features, detection_features, distance_metric="cosine"
        )
        
        # Combine distance matrices
        combined_distances = combine_distance_matrices(
            motion_distances, 
            appearance_distances,
            motion_weight=self.motion_weight,
            appearance_weight=self.appearance_weight
        )
        
        # Apply basketball-specific constraints
        combined_distances = self._apply_basketball_constraints(
            combined_distances, track_predictions, detections
        )
        
        # Solve assignment problem
        if self.assignment_algorithm == 'hungarian':
            matches, unmatched_tracks, unmatched_detections = hungarian_assignment(
                combined_distances, max_cost=self.max_cosine_distance
            )
        else:  # greedy
            from .utils import greedy_assignment
            matches, unmatched_tracks, unmatched_detections = greedy_assignment(
                combined_distances, max_cost=self.max_cosine_distance
            )
        
        return matches, unmatched_tracks, unmatched_detections
    
    def _apply_basketball_constraints(
        self, 
        distance_matrix: np.ndarray,
        track_predictions: List[Dict],
        detections: List[Dict]
    ) -> np.ndarray:
        """Apply basketball-specific constraints to distance matrix"""
        constrained_matrix = distance_matrix.copy()
        
        for t_idx, track in enumerate(track_predictions):
            for d_idx, detection in enumerate(detections):
                # Validate track-detection pair
                if not validate_track_detection_pair(track, detection):
                    constrained_matrix[t_idx, d_idx] = 2.0  # Maximum distance
                
                # Team consistency constraint
                if (self.team_consistency and 
                    'team' in track and 'team' in detection and
                    track['team'] != 'unknown' and detection['team'] != 'unknown' and
                    track['team'] != detection['team']):
                    
                    # Penalize team mismatches
                    penalty = self.config.get('basketball', {}).get('team_switching_penalty', 2.0)
                    constrained_matrix[t_idx, d_idx] = min(2.0, constrained_matrix[t_idx, d_idx] * penalty)
        
        return constrained_matrix
    
    def get_tracking_statistics(self) -> Dict:
        """Get comprehensive tracking statistics"""
        active_tracks = [t for t in self.tracks if not t.is_deleted()]
        confirmed_tracks = [t for t in active_tracks if t.is_confirmed()]
        tentative_tracks = [t for t in active_tracks if t.state == 'tentative']
        
        # Calculate average processing times
        avg_times = {}
        for component, times in self.processing_times.items():
            if times:
                avg_times[component] = {
                    'avg_ms': np.mean(times) * 1000,
                    'max_ms': np.max(times) * 1000,
                    'min_ms': np.min(times) * 1000
                }
            else:
                avg_times[component] = {'avg_ms': 0, 'max_ms': 0, 'min_ms': 0}
        
        # Team distribution
        team_distribution = defaultdict(int)
        for track in confirmed_tracks:
            team_distribution[track.team] += 1
        
        return {
            'frame_count': self.frame_count,
            'active_tracks': len(active_tracks),
            'confirmed_tracks': len(confirmed_tracks),
            'tentative_tracks': len(tentative_tracks),
            'deleted_tracks': len(self.deleted_tracks),
            'next_id': self.next_id,
            'team_distribution': dict(team_distribution),
            'processing_times': avg_times,
            'average_track_age': np.mean([t.age for t in active_tracks]) if active_tracks else 0,
            'id_retention_rate': len(confirmed_tracks) / max(1, len(active_tracks))
        }
    
    def visualize_tracks(
        self, 
        image: np.ndarray, 
        tracks: List[Dict],
        show_trails: bool = True,
        show_ids: bool = True,
        show_velocity: bool = False
    ) -> np.ndarray:
        """
        Visualize tracks on image
        
        Args:
            image: Input image
            tracks: List of track dictionaries
            show_trails: Whether to show track trails
            show_ids: Whether to show track IDs
            show_velocity: Whether to show velocity vectors
            
        Returns:
            Visualized image
        """
        vis_image = image.copy()
        
        # Color palette for different tracks
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (0, 128, 0), (128, 128, 0), (0, 128, 128), (128, 0, 0)
        ]
        
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            velocity = track['velocity']
            center = track['center']
            team = track.get('team', 'unknown')
            
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            color = colors[track_id % len(colors)]
            
            # Team-based color override
            if team == 'red':
                color = (0, 0, 255)
            elif team == 'green':
                color = (0, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID
            if show_ids:
                label = f"#{track_id}"
                if team != 'unknown':
                    label += f" ({team[0].upper()})"
                
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), color, -1)
                cv2.putText(vis_image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw center point
            cv2.circle(vis_image, (int(center[0]), int(center[1])), 4, color, -1)
            
            # Draw velocity vector
            if show_velocity:
                vx, vy = velocity
                end_x = int(center[0] + vx * 5)  # Scale velocity for visibility
                end_y = int(center[1] + vy * 5)
                cv2.arrowedLine(vis_image, (int(center[0]), int(center[1])), 
                               (end_x, end_y), color, 2, tipLength=0.3)
        
        return vis_image
    
    def reset(self):
        """Reset tracker state"""
        self.tracks = []
        self.deleted_tracks = []
        self.next_id = 1
        self.frame_count = 0
        self.processing_times = {key: [] for key in self.processing_times}
        print("DeepSORT tracker reset")
    
    def get_track_by_id(self, track_id: int) -> Optional[Dict]:
        """Get track information by ID"""
        for track in self.tracks:
            if track.track_id == track_id:
                return track.get_state()
        return None
    
    def export_tracks(self, filename: str):
        """Export track data to file"""
        import json
        
        track_data = {
            'statistics': self.get_tracking_statistics(),
            'active_tracks': [track.get_state() for track in self.tracks],
            'deleted_tracks': [track.get_state() for track in self.deleted_tracks]
        }
        
        with open(filename, 'w') as f:
            json.dump(track_data, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        
        print(f"Track data exported to {filename}")
