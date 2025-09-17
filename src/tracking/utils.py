"""Tracking Utilities for Basketball Player Tracking

This module provides utility functions for DeepSORT tracking including
data association algorithms, distance computations, and basketball-specific
tracking optimizations.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        bbox1: First bounding box [x1, y1, x2, y2]
        bbox2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Calculate intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union area
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def calculate_center_distance(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Euclidean distance between bounding box centers
    
    Args:
        bbox1: First bounding box [x1, y1, x2, y2]
        bbox2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        Distance between centers
    """
    cx1 = (bbox1[0] + bbox1[2]) / 2.0
    cy1 = (bbox1[1] + bbox1[3]) / 2.0
    cx2 = (bbox2[0] + bbox2[2]) / 2.0
    cy2 = (bbox2[1] + bbox2[3]) / 2.0
    
    return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def calculate_aspect_ratio_similarity(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate aspect ratio similarity between two bounding boxes
    
    Args:
        bbox1: First bounding box [x1, y1, x2, y2]
        bbox2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        Similarity value between 0 and 1 (1 = identical aspect ratio)
    """
    w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
    w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
    
    ar1 = w1 / h1 if h1 > 0 else 1.0
    ar2 = w2 / h2 if h2 > 0 else 1.0
    
    # Calculate similarity (inverse of ratio difference)
    ratio_diff = abs(ar1 - ar2) / max(ar1, ar2)
    similarity = 1.0 / (1.0 + ratio_diff)
    
    return similarity


def calculate_size_similarity(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate size similarity between two bounding boxes
    
    Args:
        bbox1: First bounding box [x1, y1, x2, y2]
        bbox2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        Similarity value between 0 and 1
    """
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    if area1 == 0 or area2 == 0:
        return 0.0
    
    # Calculate similarity as overlap of smaller to larger
    smaller_area = min(area1, area2)
    larger_area = max(area1, area2)
    
    return smaller_area / larger_area


def hungarian_assignment(cost_matrix: np.ndarray, max_cost: float = 1.0) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Solve assignment problem using Hungarian algorithm
    
    Args:
        cost_matrix: Cost matrix [tracks, detections]
        max_cost: Maximum allowed cost for assignment
        
    Returns:
        Tuple of (matches, unmatched_tracks, unmatched_detections)
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
    
    # Solve assignment problem
    track_indices, detection_indices = linear_sum_assignment(cost_matrix)
    
    matches = []
    unmatched_tracks = []
    unmatched_detections = []
    
    # Process assignments
    for track_idx, det_idx in zip(track_indices, detection_indices):
        if cost_matrix[track_idx, det_idx] <= max_cost:
            matches.append((track_idx, det_idx))
        else:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(det_idx)
    
    # Find unmatched tracks and detections
    matched_track_indices = set(match[0] for match in matches)
    matched_detection_indices = set(match[1] for match in matches)
    
    for track_idx in range(cost_matrix.shape[0]):
        if track_idx not in matched_track_indices:
            unmatched_tracks.append(track_idx)
    
    for det_idx in range(cost_matrix.shape[1]):
        if det_idx not in matched_detection_indices:
            unmatched_detections.append(det_idx)
    
    return matches, unmatched_tracks, unmatched_detections


def greedy_assignment(cost_matrix: np.ndarray, max_cost: float = 1.0) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Solve assignment problem using greedy algorithm (faster but less optimal)
    
    Args:
        cost_matrix: Cost matrix [tracks, detections]
        max_cost: Maximum allowed cost for assignment
        
    Returns:
        Tuple of (matches, unmatched_tracks, unmatched_detections)
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
    
    matches = []
    used_tracks = set()
    used_detections = set()
    
    # Create list of (cost, track_idx, det_idx) sorted by cost
    assignments = []
    for track_idx in range(cost_matrix.shape[0]):
        for det_idx in range(cost_matrix.shape[1]):
            if cost_matrix[track_idx, det_idx] <= max_cost:
                assignments.append((cost_matrix[track_idx, det_idx], track_idx, det_idx))
    
    # Sort by cost (lowest first)
    assignments.sort(key=lambda x: x[0])
    
    # Greedily assign lowest cost pairs
    for cost, track_idx, det_idx in assignments:
        if track_idx not in used_tracks and det_idx not in used_detections:
            matches.append((track_idx, det_idx))
            used_tracks.add(track_idx)
            used_detections.add(det_idx)
    
    # Find unmatched tracks and detections
    unmatched_tracks = [i for i in range(cost_matrix.shape[0]) if i not in used_tracks]
    unmatched_detections = [i for i in range(cost_matrix.shape[1]) if i not in used_detections]
    
    return matches, unmatched_tracks, unmatched_detections


def compute_motion_distance_matrix(
    track_predictions: List[Dict],
    detections: List[Dict],
    distance_type: str = "iou"
) -> np.ndarray:
    """
    Compute distance matrix based on motion/geometry
    
    Args:
        track_predictions: List of track predictions with 'bbox' key
        detections: List of detections with 'bbox' key
        distance_type: Type of distance ('iou', 'center', 'combined')
        
    Returns:
        Distance matrix [tracks, detections]
    """
    if not track_predictions or not detections:
        return np.full((len(track_predictions), len(detections)), 1.0)
    
    num_tracks = len(track_predictions)
    num_detections = len(detections)
    distance_matrix = np.zeros((num_tracks, num_detections))
    
    for t_idx, track in enumerate(track_predictions):
        track_bbox = track['bbox']
        
        for d_idx, detection in enumerate(detections):
            det_bbox = detection['bbox']
            
            if distance_type == "iou":
                iou = calculate_iou(track_bbox, det_bbox)
                distance_matrix[t_idx, d_idx] = 1.0 - iou
            
            elif distance_type == "center":
                center_dist = calculate_center_distance(track_bbox, det_bbox)
                # Normalize by image diagonal (assuming 1920x1080 max)
                max_dist = np.sqrt(1920**2 + 1080**2)
                distance_matrix[t_idx, d_idx] = min(center_dist / max_dist, 1.0)
            
            elif distance_type == "combined":
                iou = calculate_iou(track_bbox, det_bbox)
                center_dist = calculate_center_distance(track_bbox, det_bbox)
                max_dist = np.sqrt(1920**2 + 1080**2)
                
                iou_dist = 1.0 - iou
                center_dist_norm = min(center_dist / max_dist, 1.0)
                
                # Weighted combination
                distance_matrix[t_idx, d_idx] = 0.7 * iou_dist + 0.3 * center_dist_norm
    
    return distance_matrix


def compute_appearance_distance_matrix(
    track_features: List[np.ndarray],
    detection_features: List[np.ndarray],
    distance_metric: str = "cosine"
) -> np.ndarray:
    """
    Compute distance matrix based on appearance features
    
    Args:
        track_features: List of track feature vectors
        detection_features: List of detection feature vectors
        distance_metric: Distance metric ('cosine', 'euclidean', 'manhattan')
        
    Returns:
        Distance matrix [tracks, detections]
    """
    if not track_features or not detection_features:
        return np.full((len(track_features), len(detection_features)), 1.0)
    
    # Stack features into matrices
    track_matrix = np.stack(track_features)  # [num_tracks, feature_dim]
    detection_matrix = np.stack(detection_features)  # [num_detections, feature_dim]
    
    if distance_metric == "cosine":
        # Cosine distance (1 - cosine similarity)
        # For L2-normalized features, cosine similarity = dot product
        similarities = np.dot(track_matrix, detection_matrix.T)
        distances = 1.0 - similarities
        distances = np.clip(distances, 0.0, 2.0)
    
    elif distance_metric == "euclidean":
        distances = cdist(track_matrix, detection_matrix, metric='euclidean')
        # Normalize to [0, 2] range (max distance between unit vectors)
        distances = distances / np.sqrt(2)
        distances = np.clip(distances, 0.0, 2.0)
    
    elif distance_metric == "manhattan":
        distances = cdist(track_matrix, detection_matrix, metric='manhattan')
        # Normalize to [0, 2] range
        distances = distances / 2.0
        distances = np.clip(distances, 0.0, 2.0)
    
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")
    
    return distances


def combine_distance_matrices(
    motion_distances: np.ndarray,
    appearance_distances: np.ndarray,
    motion_weight: float = 0.3,
    appearance_weight: float = 0.7
) -> np.ndarray:
    """
    Combine motion and appearance distance matrices
    
    Args:
        motion_distances: Motion-based distance matrix
        appearance_distances: Appearance-based distance matrix
        motion_weight: Weight for motion distances
        appearance_weight: Weight for appearance distances
        
    Returns:
        Combined distance matrix
    """
    if motion_distances.shape != appearance_distances.shape:
        raise ValueError("Distance matrices must have the same shape")
    
    # Ensure weights sum to 1
    total_weight = motion_weight + appearance_weight
    motion_weight /= total_weight
    appearance_weight /= total_weight
    
    combined_distances = (motion_weight * motion_distances + 
                         appearance_weight * appearance_distances)
    
    return combined_distances


def filter_detections_by_tracking_context(
    detections: List[Dict],
    active_tracks: List[Dict],
    image_shape: Tuple[int, int],
    min_distance_threshold: float = 50.0
) -> List[Dict]:
    """
    Filter detections based on tracking context (basketball-specific)
    
    Args:
        detections: List of detections
        active_tracks: List of active tracks
        image_shape: Image shape (height, width)
        min_distance_threshold: Minimum distance between detections
        
    Returns:
        Filtered detections
    """
    if not detections:
        return detections
    
    filtered_detections = []
    
    for detection in detections:
        det_bbox = detection['bbox']
        det_center = [(det_bbox[0] + det_bbox[2]) / 2, (det_bbox[1] + det_bbox[3]) / 2]
        
        # Check if detection is too close to existing tracks
        too_close_to_track = False
        for track in active_tracks:
            track_bbox = track['bbox']
            track_center = [(track_bbox[0] + track_bbox[2]) / 2, (track_bbox[1] + track_bbox[3]) / 2]
            
            distance = np.sqrt((det_center[0] - track_center[0])**2 + 
                             (det_center[1] - track_center[1])**2)
            
            if distance < min_distance_threshold:
                too_close_to_track = True
                break
        
        if not too_close_to_track:
            filtered_detections.append(detection)
    
    return filtered_detections


def validate_track_detection_pair(
    track: Dict,
    detection: Dict,
    max_velocity: float = 200.0,
    max_size_change: float = 0.5
) -> bool:
    """
    Validate if a track-detection pair is reasonable (basketball-specific)
    
    Args:
        track: Track information with 'bbox' and 'velocity'
        detection: Detection information with 'bbox'
        max_velocity: Maximum allowed velocity (pixels/frame)
        max_size_change: Maximum allowed size change ratio
        
    Returns:
        True if pair is valid
    """
    track_bbox = track['bbox']
    det_bbox = detection['bbox']
    
    # Check velocity constraint
    if 'velocity' in track:
        vx, vy = track['velocity']
        velocity_magnitude = np.sqrt(vx**2 + vy**2)
        
        if velocity_magnitude > max_velocity:
            return False
    
    # Check size change constraint
    track_area = (track_bbox[2] - track_bbox[0]) * (track_bbox[3] - track_bbox[1])
    det_area = (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1])
    
    if track_area > 0:
        size_change_ratio = abs(det_area - track_area) / track_area
        if size_change_ratio > max_size_change:
            return False
    
    # Check aspect ratio constraint
    track_ar = (track_bbox[2] - track_bbox[0]) / (track_bbox[3] - track_bbox[1])
    det_ar = (det_bbox[2] - det_bbox[0]) / (det_bbox[3] - det_bbox[1])
    
    ar_change_ratio = abs(det_ar - track_ar) / max(track_ar, det_ar)
    if ar_change_ratio > 0.3:  # 30% aspect ratio change threshold
        return False
    
    return True


def non_maximum_suppression_tracks(
    tracks: List[Dict],
    iou_threshold: float = 0.3
) -> List[Dict]:
    """
    Apply NMS to remove duplicate tracks
    
    Args:
        tracks: List of tracks with 'bbox' and 'confidence' or 'hits'
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Filtered tracks
    """
    if not tracks:
        return tracks
    
    # Sort by confidence/hits (higher first)
    sorted_tracks = sorted(tracks, key=lambda x: x.get('hits', x.get('confidence', 0)), reverse=True)
    
    keep = []
    while sorted_tracks:
        current = sorted_tracks.pop(0)
        keep.append(current)
        
        # Remove tracks with high IoU overlap
        remaining_tracks = []
        for track in sorted_tracks:
            iou = calculate_iou(current['bbox'], track['bbox'])
            if iou < iou_threshold:
                remaining_tracks.append(track)
        
        sorted_tracks = remaining_tracks
    
    return keep


def smooth_track_trajectory(
    track_history: List[List[float]],
    window_size: int = 5,
    smoothing_factor: float = 0.3
) -> List[float]:
    """
    Smooth track trajectory using moving average
    
    Args:
        track_history: List of bounding boxes [[x1, y1, x2, y2], ...]
        window_size: Size of smoothing window
        smoothing_factor: Smoothing strength (0 = no smoothing, 1 = maximum smoothing)
        
    Returns:
        Smoothed bounding box [x1, y1, x2, y2]
    """
    if not track_history:
        return [0, 0, 0, 0]
    
    if len(track_history) == 1:
        return track_history[0]
    
    # Use recent history for smoothing
    recent_history = track_history[-window_size:]
    
    # Calculate weighted average (more weight to recent positions)
    weights = np.exp(np.linspace(-1, 0, len(recent_history)))
    weights = weights / weights.sum()
    
    smoothed_bbox = np.zeros(4)
    for i, bbox in enumerate(recent_history):
        smoothed_bbox += weights[i] * np.array(bbox)
    
    # Blend with current position
    current_bbox = np.array(track_history[-1])
    final_bbox = (1 - smoothing_factor) * current_bbox + smoothing_factor * smoothed_bbox
    
    return final_bbox.tolist()


def calculate_tracking_metrics(
    ground_truth_tracks: List[Dict],
    predicted_tracks: List[Dict],
    iou_threshold: float = 0.5
) -> Dict:
    """
    Calculate tracking performance metrics
    
    Args:
        ground_truth_tracks: Ground truth tracks
        predicted_tracks: Predicted tracks
        iou_threshold: IoU threshold for matching
        
    Returns:
        Dictionary with tracking metrics
    """
    if not ground_truth_tracks or not predicted_tracks:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'id_switches': 0,
            'track_purity': 0.0
        }
    
    # Simple implementation - in practice, this would be more complex
    # using metrics like MOTA, MOTP, IDF1, etc.
    
    # Calculate IoU-based matches
    matches = 0
    for gt_track in ground_truth_tracks:
        for pred_track in predicted_tracks:
            iou = calculate_iou(gt_track['bbox'], pred_track['bbox'])
            if iou >= iou_threshold:
                matches += 1
                break
    
    precision = matches / len(predicted_tracks) if predicted_tracks else 0
    recall = matches / len(ground_truth_tracks) if ground_truth_tracks else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'matches': matches,
        'total_predictions': len(predicted_tracks),
        'total_ground_truth': len(ground_truth_tracks)
    }
