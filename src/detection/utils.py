"""Detection utilities for basketball player detection

This module provides utility functions for YOLO detection processing,
including non-maximum suppression, bbox operations, and validation.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
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


def non_maximum_suppression(detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """
    Apply Non-Maximum Suppression to remove overlapping detections
    
    Args:
        detections: List of detection dictionaries
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Filtered list of detections
    """
    if not detections:
        return []
    
    # Sort by confidence (highest first)
    sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    while sorted_detections:
        # Keep the highest confidence detection
        current = sorted_detections.pop(0)
        keep.append(current)
        
        # Remove detections with high IoU overlap
        filtered_detections = []
        for det in sorted_detections:
            iou = calculate_iou(current['bbox'], det['bbox'])
            if iou < iou_threshold:
                filtered_detections.append(det)
        
        sorted_detections = filtered_detections
    
    return keep


def filter_detections_by_size(
    detections: List[Dict], 
    min_area: int = 1000, 
    max_area: int = 50000,
    min_aspect_ratio: float = 0.3,
    max_aspect_ratio: float = 1.0
) -> List[Dict]:
    """
    Filter detections based on size and aspect ratio constraints
    
    Args:
        detections: List of detection dictionaries
        min_area: Minimum bounding box area
        max_area: Maximum bounding box area
        min_aspect_ratio: Minimum width/height ratio
        max_aspect_ratio: Maximum width/height ratio
        
    Returns:
        Filtered list of detections
    """
    filtered = []
    
    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = width / height if height > 0 else 0
        
        # Check constraints
        if (min_area <= area <= max_area and 
            min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
            filtered.append(det)
    
    return filtered


def filter_detections_by_region(
    detections: List[Dict], 
    image_shape: Tuple[int, int], 
    court_mask: Optional[np.ndarray] = None,
    margin_ratio: float = 0.05
) -> List[Dict]:
    """
    Filter detections based on court region or image margins
    
    Args:
        detections: List of detection dictionaries
        image_shape: Image shape (height, width)
        court_mask: Optional court mask (white pixels = valid region)
        margin_ratio: Margin ratio from image edges to exclude
        
    Returns:
        Filtered list of detections
    """
    filtered = []
    height, width = image_shape[:2]
    
    # Calculate margins
    margin_x = int(width * margin_ratio)
    margin_y = int(height * margin_ratio)
    
    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = bbox
        center_x, center_y = det['center']
        
        # Check if detection is within image margins
        if (margin_x <= center_x <= width - margin_x and 
            margin_y <= center_y <= height - margin_y):
            
            # If court mask is provided, check if center is on court
            if court_mask is not None:
                if (0 <= center_y < court_mask.shape[0] and 
                    0 <= center_x < court_mask.shape[1]):
                    if court_mask[center_y, center_x] > 0:  # White pixel = on court
                        filtered.append(det)
                # Skip if center is outside court
            else:
                # No court mask, just use margin filtering
                filtered.append(det)
    
    return filtered


def merge_close_detections(
    detections: List[Dict], 
    distance_threshold: float = 50.0
) -> List[Dict]:
    """
    Merge detections that are very close to each other (likely same player)
    
    Args:
        detections: List of detection dictionaries
        distance_threshold: Maximum distance between centers to merge
        
    Returns:
        List of merged detections
    """
    if not detections:
        return []
    
    merged = []
    used = set()
    
    for i, det1 in enumerate(detections):
        if i in used:
            continue
        
        # Find all detections close to this one
        group = [det1]
        center1 = np.array(det1['center'])
        
        for j, det2 in enumerate(detections[i+1:], i+1):
            if j in used:
                continue
            
            center2 = np.array(det2['center'])
            distance = np.linalg.norm(center1 - center2)
            
            if distance < distance_threshold:
                group.append(det2)
                used.add(j)
        
        # Merge the group (keep highest confidence, average position)
        if len(group) == 1:
            merged.append(group[0])
        else:
            # Sort by confidence and take the best one as base
            group.sort(key=lambda x: x['confidence'], reverse=True)
            best_det = group[0].copy()
            
            # Average the positions
            avg_center_x = int(np.mean([d['center'][0] for d in group]))
            avg_center_y = int(np.mean([d['center'][1] for d in group]))
            
            # Update the merged detection
            best_det['center'] = [avg_center_x, avg_center_y]
            best_det['confidence'] = max(d['confidence'] for d in group)
            
            merged.append(best_det)
        
        used.add(i)
    
    return merged


def validate_detection_sequence(
    current_detections: List[Dict],
    previous_detections: List[Dict],
    max_movement: float = 100.0,
    confidence_boost: float = 0.1
) -> List[Dict]:
    """
    Validate detections based on previous frame (temporal consistency)
    
    Args:
        current_detections: Current frame detections
        previous_detections: Previous frame detections
        max_movement: Maximum allowed movement between frames
        confidence_boost: Confidence boost for temporally consistent detections
        
    Returns:
        Validated and adjusted detections
    """
    if not previous_detections:
        return current_detections
    
    validated = []
    
    for curr_det in current_detections:
        curr_center = np.array(curr_det['center'])
        min_distance = float('inf')
        
        # Find closest previous detection
        for prev_det in previous_detections:
            prev_center = np.array(prev_det['center'])
            distance = np.linalg.norm(curr_center - prev_center)
            min_distance = min(min_distance, distance)
        
        # Adjust confidence based on temporal consistency
        adjusted_det = curr_det.copy()
        if min_distance < max_movement:
            # Boost confidence for consistent detections
            adjusted_det['confidence'] = min(1.0, curr_det['confidence'] + confidence_boost)
            adjusted_det['temporal_consistency'] = True
        else:
            # Reduce confidence for inconsistent detections
            adjusted_det['confidence'] = max(0.0, curr_det['confidence'] - confidence_boost)
            adjusted_det['temporal_consistency'] = False
        
        validated.append(adjusted_det)
    
    return validated


def draw_detection_info(
    image: np.ndarray, 
    detections: List[Dict],
    show_stats: bool = True
) -> np.ndarray:
    """
    Draw comprehensive detection information on image
    
    Args:
        image: Input image
        detections: List of detections
        show_stats: Whether to show detection statistics
        
    Returns:
        Image with detection information
    """
    vis_image = image.copy()
    
    if not detections:
        if show_stats:
            cv2.putText(vis_image, "No players detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return vis_image
    
    # Draw individual detections
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        center = det['center']
        
        # Color coding based on confidence
        if conf > 0.8:
            color = (0, 255, 0)  # High confidence - Green
        elif conf > 0.5:
            color = (0, 255, 255)  # Medium confidence - Yellow
        else:
            color = (0, 0, 255)  # Low confidence - Red
        
        # Draw bounding box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw center point
        cv2.circle(vis_image, tuple(center), 4, color, -1)
        
        # Draw detection number and confidence
        label = f"{i+1}: {conf:.2f}"
        cv2.putText(vis_image, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw statistics
    if show_stats:
        stats_y = 30
        cv2.putText(vis_image, f"Players: {len(detections)}", (10, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if detections:
            avg_conf = np.mean([d['confidence'] for d in detections])
            cv2.putText(vis_image, f"Avg Conf: {avg_conf:.2f}", (10, stats_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return vis_image


def create_detection_heatmap(
    image_shape: Tuple[int, int], 
    detections_history: List[List[Dict]],
    sigma: float = 20.0
) -> np.ndarray:
    """
    Create a heatmap of player positions over time
    
    Args:
        image_shape: Shape of the image (height, width)
        detections_history: List of detection lists from multiple frames
        sigma: Gaussian blur sigma for heatmap smoothing
        
    Returns:
        Heatmap as grayscale image
    """
    height, width = image_shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    # Accumulate all detection centers
    for detections in detections_history:
        for det in detections:
            center_x, center_y = det['center']
            if 0 <= center_x < width and 0 <= center_y < height:
                # Weight by confidence
                weight = det['confidence']
                heatmap[center_y, center_x] += weight
    
    # Apply Gaussian blur for smooth heatmap
    if sigma > 0:
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigma)
    
    # Normalize to 0-255 range
    if heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
    
    return heatmap
