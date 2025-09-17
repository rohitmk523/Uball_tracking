"""Kalman Filter for Basketball Player Motion Prediction

This module implements a Kalman filter optimized for basketball player tracking,
handling the specific motion patterns and constraints of basketball gameplay.
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from typing import List, Tuple, Optional


class BasketballKalmanFilter:
    """
    Kalman filter for basketball player tracking with motion prediction
    
    State vector: [x, y, w, h, vx, vy, vw, vh]
    - (x, y): center position
    - (w, h): width and height of bounding box
    - (vx, vy): velocity of center
    - (vw, vh): velocity of size change
    """
    
    def __init__(self, initial_bbox: List[float], dt: float = 1.0):
        """
        Initialize Kalman filter for basketball player tracking
        
        Args:
            initial_bbox: Initial bounding box [x1, y1, x2, y2]
            dt: Time step between frames
        """
        self.dt = dt
        
        # Convert bbox to center format
        x1, y1, x2, y2 = initial_bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        
        # Initialize Kalman filter (8 states, 4 measurements)
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # Initial state [cx, cy, w, h, vx, vy, vw, vh]
        self.kf.x = np.array([cx, cy, w, h, 0, 0, 0, 0])
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, dt, 0,  0,  0],   # x = x + vx*dt
            [0, 1, 0, 0, 0,  dt, 0,  0],   # y = y + vy*dt
            [0, 0, 1, 0, 0,  0,  dt, 0],   # w = w + vw*dt
            [0, 0, 0, 1, 0,  0,  0,  dt],  # h = h + vh*dt
            [0, 0, 0, 0, 1,  0,  0,  0],   # vx = vx
            [0, 0, 0, 0, 0,  1,  0,  0],   # vy = vy
            [0, 0, 0, 0, 0,  0,  1,  0],   # vw = vw
            [0, 0, 0, 0, 0,  0,  0,  1]    # vh = vh
        ])
        
        # Measurement matrix (we observe position and size)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],   # measure x
            [0, 1, 0, 0, 0, 0, 0, 0],   # measure y
            [0, 0, 1, 0, 0, 0, 0, 0],   # measure w
            [0, 0, 0, 1, 0, 0, 0, 0]    # measure h
        ])
        
        # Process noise covariance (basketball-specific tuning)
        self._setup_process_noise()
        
        # Measurement noise covariance
        self._setup_measurement_noise()
        
        # Initial covariance
        self._setup_initial_covariance()
        
        # Basketball-specific constraints
        self.max_velocity = 200.0  # pixels per frame
        self.max_size_change = 0.5  # maximum size change ratio per frame
        
        # Track statistics
        self.age = 0
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0
    
    def _setup_process_noise(self):
        """Setup process noise covariance matrix for basketball tracking"""
        # Basketball players have different motion characteristics
        position_noise = 4.0  # Position uncertainty
        velocity_noise = 10.0  # Velocity uncertainty
        size_noise = 1.0      # Size uncertainty
        size_vel_noise = 2.0  # Size change uncertainty
        
        self.kf.Q = np.diag([
            position_noise,    # x
            position_noise,    # y
            size_noise,        # w
            size_noise,        # h
            velocity_noise,    # vx
            velocity_noise,    # vy
            size_vel_noise,    # vw
            size_vel_noise     # vh
        ])
    
    def _setup_measurement_noise(self):
        """Setup measurement noise covariance matrix"""
        # YOLO detection uncertainty
        detection_noise = 10.0
        size_noise = 5.0
        
        self.kf.R = np.diag([
            detection_noise,   # x measurement noise
            detection_noise,   # y measurement noise
            size_noise,        # w measurement noise
            size_noise         # h measurement noise
        ])
    
    def _setup_initial_covariance(self):
        """Setup initial state covariance matrix"""
        self.kf.P = np.diag([
            100,   # x initial uncertainty
            100,   # y initial uncertainty
            100,   # w initial uncertainty
            100,   # h initial uncertainty
            1000,  # vx initial uncertainty
            1000,  # vy initial uncertainty
            100,   # vw initial uncertainty
            100    # vh initial uncertainty
        ])
    
    def predict(self) -> np.ndarray:
        """
        Predict next state using motion model
        
        Returns:
            Predicted state vector
        """
        # Standard Kalman prediction
        self.kf.predict()
        
        # Apply basketball-specific constraints
        self._apply_motion_constraints()
        
        # Update tracking statistics
        self.age += 1
        self.time_since_update += 1
        
        return self.kf.x.copy()
    
    def update(self, bbox: List[float]):
        """
        Update filter with new measurement
        
        Args:
            bbox: Observed bounding box [x1, y1, x2, y2]
        """
        # Convert bbox to center format
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        
        # Measurement vector
        measurement = np.array([cx, cy, w, h])
        
        # Kalman update
        self.kf.update(measurement)
        
        # Apply constraints after update
        self._apply_state_constraints()
        
        # Update tracking statistics
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
    
    def _apply_motion_constraints(self):
        """Apply basketball-specific motion constraints"""
        # Limit velocity to reasonable values
        self.kf.x[4] = np.clip(self.kf.x[4], -self.max_velocity, self.max_velocity)  # vx
        self.kf.x[5] = np.clip(self.kf.x[5], -self.max_velocity, self.max_velocity)  # vy
        
        # Limit size change velocity
        max_size_vel = self.max_size_change * max(self.kf.x[2], self.kf.x[3])
        self.kf.x[6] = np.clip(self.kf.x[6], -max_size_vel, max_size_vel)  # vw
        self.kf.x[7] = np.clip(self.kf.x[7], -max_size_vel, max_size_vel)  # vh
    
    def _apply_state_constraints(self):
        """Apply constraints to state variables"""
        # Ensure positive dimensions
        self.kf.x[2] = max(self.kf.x[2], 10)  # minimum width
        self.kf.x[3] = max(self.kf.x[3], 20)  # minimum height
        
        # Basketball player aspect ratio constraints
        aspect_ratio = self.kf.x[2] / self.kf.x[3]
        if aspect_ratio > 1.2:  # too wide
            self.kf.x[2] = self.kf.x[3] * 1.2
        elif aspect_ratio < 0.3:  # too narrow
            self.kf.x[2] = self.kf.x[3] * 0.3
    
    def get_state(self) -> List[float]:
        """
        Get current state as bounding box
        
        Returns:
            Bounding box [x1, y1, x2, y2]
        """
        cx, cy, w, h = self.kf.x[0], self.kf.x[1], self.kf.x[2], self.kf.x[3]
        
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        
        return [float(x1), float(y1), float(x2), float(y2)]
    
    def get_velocity(self) -> Tuple[float, float]:
        """
        Get current velocity
        
        Returns:
            Tuple of (vx, vy) velocity components
        """
        return float(self.kf.x[4]), float(self.kf.x[5])
    
    def get_predicted_position(self, frames_ahead: int = 1) -> List[float]:
        """
        Get predicted position N frames ahead
        
        Args:
            frames_ahead: Number of frames to predict ahead
            
        Returns:
            Predicted bounding box [x1, y1, x2, y2]
        """
        # Create temporary state for prediction
        future_state = self.kf.x.copy()
        
        # Apply motion model for N frames
        for _ in range(frames_ahead):
            future_state[0] += future_state[4] * self.dt  # x += vx * dt
            future_state[1] += future_state[5] * self.dt  # y += vy * dt
            future_state[2] += future_state[6] * self.dt  # w += vw * dt
            future_state[3] += future_state[7] * self.dt  # h += vh * dt
        
        # Convert to bbox format
        cx, cy, w, h = future_state[0], future_state[1], future_state[2], future_state[3]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        
        return [float(x1), float(y1), float(x2), float(y2)]
    
    def get_state_uncertainty(self) -> float:
        """
        Get current state uncertainty (trace of position covariance)
        
        Returns:
            Uncertainty measure
        """
        # Sum of position variances
        return float(self.kf.P[0, 0] + self.kf.P[1, 1])
    
    def is_stable(self, min_hits: int = 3, max_time_since_update: int = 5) -> bool:
        """
        Check if track is stable enough for output
        
        Args:
            min_hits: Minimum number of hits required
            max_time_since_update: Maximum frames since last update
            
        Returns:
            True if track is stable
        """
        return (self.hits >= min_hits and 
                self.time_since_update <= max_time_since_update)
    
    def should_delete(self, max_age: int = 30, max_time_since_update: int = 10) -> bool:
        """
        Check if track should be deleted
        
        Args:
            max_age: Maximum age before deletion
            max_time_since_update: Maximum frames without update
            
        Returns:
            True if track should be deleted
        """
        return (self.age > max_age or 
                self.time_since_update > max_time_since_update)
    
    def get_track_info(self) -> dict:
        """
        Get comprehensive track information
        
        Returns:
            Dictionary with track statistics and state
        """
        bbox = self.get_state()
        velocity = self.get_velocity()
        
        return {
            'bbox': bbox,
            'velocity': velocity,
            'age': self.age,
            'hits': self.hits,
            'hit_streak': self.hit_streak,
            'time_since_update': self.time_since_update,
            'uncertainty': self.get_state_uncertainty(),
            'is_stable': self.is_stable(),
            'should_delete': self.should_delete(),
            'center': [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
            'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        }


class KalmanTrackManager:
    """Manager for multiple Kalman filter tracks"""
    
    def __init__(self):
        """Initialize track manager"""
        self.tracks = []
        self.next_id = 0
        self.deleted_tracks = []
        
        # Basketball-specific parameters
        self.max_tracks = 15  # Maximum players on court
        self.cleanup_interval = 30  # Frames between cleanup
        self.frame_count = 0
    
    def create_track(self, bbox: List[float]) -> int:
        """
        Create new track
        
        Args:
            bbox: Initial bounding box
            
        Returns:
            Track ID
        """
        if len(self.tracks) >= self.max_tracks:
            # Remove oldest unstable track
            self._cleanup_tracks(force=True)
        
        track_id = self.next_id
        self.next_id += 1
        
        track_info = {
            'id': track_id,
            'filter': BasketballKalmanFilter(bbox),
            'created_frame': self.frame_count
        }
        
        self.tracks.append(track_info)
        return track_id
    
    def update_track(self, track_id: int, bbox: List[float]) -> bool:
        """
        Update existing track
        
        Args:
            track_id: ID of track to update
            bbox: New bounding box measurement
            
        Returns:
            True if track was found and updated
        """
        for track in self.tracks:
            if track['id'] == track_id:
                track['filter'].update(bbox)
                return True
        return False
    
    def predict_all(self):
        """Predict next state for all tracks"""
        for track in self.tracks:
            track['filter'].predict()
        
        self.frame_count += 1
        
        # Periodic cleanup
        if self.frame_count % self.cleanup_interval == 0:
            self._cleanup_tracks()
    
    def get_track_predictions(self) -> List[dict]:
        """
        Get predictions for all stable tracks
        
        Returns:
            List of track predictions with IDs
        """
        predictions = []
        
        for track in self.tracks:
            if track['filter'].is_stable():
                info = track['filter'].get_track_info()
                info['track_id'] = track['id']
                info['created_frame'] = track['created_frame']
                predictions.append(info)
        
        return predictions
    
    def _cleanup_tracks(self, force: bool = False):
        """Clean up old or unstable tracks"""
        tracks_to_remove = []
        
        for track in self.tracks:
            if track['filter'].should_delete() or force:
                tracks_to_remove.append(track)
        
        # Remove tracks marked for deletion
        for track in tracks_to_remove:
            self.deleted_tracks.append({
                'id': track['id'],
                'deleted_frame': self.frame_count,
                'final_info': track['filter'].get_track_info()
            })
            self.tracks.remove(track)
        
        # If force cleanup and still too many tracks, remove oldest
        if force and len(self.tracks) >= self.max_tracks:
            oldest_track = min(self.tracks, key=lambda t: t['created_frame'])
            self.tracks.remove(oldest_track)
    
    def get_statistics(self) -> dict:
        """Get tracking statistics"""
        stable_tracks = [t for t in self.tracks if t['filter'].is_stable()]
        
        return {
            'total_tracks': len(self.tracks),
            'stable_tracks': len(stable_tracks),
            'deleted_tracks': len(self.deleted_tracks),
            'next_id': self.next_id,
            'frame_count': self.frame_count,
            'average_track_age': np.mean([t['filter'].age for t in self.tracks]) if self.tracks else 0
        }
