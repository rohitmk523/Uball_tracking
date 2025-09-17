"""YOLO Detector for Basketball Player Detection

This module implements YOLOv11 Nano detector optimized for basketball players
from elevated camera angles with multi-player scenarios.
"""

import torch
from ultralytics import YOLO
import cv2
import numpy as np
import yaml
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class YOLODetector:
    """YOLOv11 Nano detector optimized for basketball players"""
    
    def __init__(
        self, 
        model_path: str = "yolo11n.pt", 
        config_path: Optional[str] = None,
        conf_threshold: float = 0.5
    ):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model weights
            config_path: Path to YOLO configuration file
            conf_threshold: Confidence threshold for detections
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        
        # Load configuration
        if config_path:
            self.config = self._load_config(config_path)
            self.conf_threshold = self.config.get('detection', {}).get(
                'confidence_threshold', conf_threshold
            )
        else:
            self.config = self._default_config()
        
        # Initialize device
        self.device = self._get_device()
        
        # Load YOLO model
        self.model = YOLO(model_path)
        
        # Basketball-specific parameters
        self.min_player_area = self.config.get('basketball', {}).get('min_player_area', 1000)
        self.max_player_area = self.config.get('basketball', {}).get('max_player_area', 50000)
        self.min_aspect_ratio = self.config.get('basketball', {}).get('min_aspect_ratio', 0.3)
        self.max_aspect_ratio = self.config.get('basketball', {}).get('max_aspect_ratio', 1.0)
        
        print(f"YOLO Detector initialized on device: {self.device}")
        print(f"Model: {model_path}")
        print(f"Confidence threshold: {self.conf_threshold}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found. Using default config.")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default configuration for YOLO detector"""
        return {
            'detection': {
                'confidence_threshold': 0.5,
                'nms_threshold': 0.4,
                'max_detections': 20,
                'classes': [0]
            },
            'basketball': {
                'min_player_area': 1000,
                'max_player_area': 50000,
                'min_aspect_ratio': 0.3,
                'max_aspect_ratio': 1.0
            }
        }
    
    def _get_device(self) -> torch.device:
        """Determine the best available device"""
        device_config = self.config.get('model', {}).get('device', 'auto')
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device_config)
    
    def _is_valid_player_detection(self, bbox: List[int]) -> bool:
        """
        Validate if detection meets basketball player criteria
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            True if detection is valid player detection
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = width / height if height > 0 else 0
        
        # Check area constraints
        if area < self.min_player_area or area > self.max_player_area:
            return False
        
        # Check aspect ratio (players should be taller than wide)
        if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
            return False
        
        return True
    
    def detect_players(self, image: np.ndarray) -> List[Dict]:
        """
        Detect basketball players in image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of detections with bounding boxes and confidence scores
        """
        # Get detection parameters from config
        detection_config = self.config.get('detection', {})
        classes = detection_config.get('classes', [0])
        nms_threshold = detection_config.get('nms_threshold', 0.4)
        max_detections = detection_config.get('max_detections', 20)
        
        # Run YOLO inference
        results = self.model(
            image, 
            conf=self.conf_threshold,
            iou=nms_threshold,
            classes=classes,
            max_det=max_detections,
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract box coordinates and confidence
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    
                    bbox = [x1, y1, x2, y2]
                    
                    # Validate detection
                    if not self._is_valid_player_detection(bbox):
                        continue
                    
                    detection = {
                        'bbox': bbox,
                        'confidence': conf,
                        'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                        'area': int((x2 - x1) * (y2 - y1)),
                        'width': int(x2 - x1),
                        'height': int(y2 - y1)
                    }
                    detections.append(detection)
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detections
    
    def visualize_detections(
        self, 
        image: np.ndarray, 
        detections: List[Dict],
        show_confidence: bool = True,
        show_center: bool = True
    ) -> np.ndarray:
        """
        Draw bounding boxes on image
        
        Args:
            image: Input image
            detections: List of detections from detect_players()
            show_confidence: Whether to show confidence scores
            show_center: Whether to show center points
            
        Returns:
            Image with drawn bounding boxes
        """
        vis_image = image.copy()
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            center = det['center']
            
            # Color based on confidence (green for high, yellow for medium, red for low)
            if conf > 0.8:
                color = (0, 255, 0)  # Green
            elif conf > 0.5:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence score and player number
            if show_confidence:
                label = f"Player {i+1}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Draw label background
                cv2.rectangle(
                    vis_image, 
                    (x1, y1 - label_size[1] - 10), 
                    (x1 + label_size[0], y1), 
                    color, 
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    vis_image, 
                    label, 
                    (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    2
                )
            
            # Draw center point
            if show_center:
                cv2.circle(vis_image, tuple(center), 3, color, -1)
        
        # Add detection statistics
        stats_text = f"Players detected: {len(detections)}"
        cv2.putText(
            vis_image, 
            stats_text, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
        return vis_image
    
    def get_detection_statistics(self, detections: List[Dict]) -> Dict:
        """
        Get statistics about current detections
        
        Args:
            detections: List of detections
            
        Returns:
            Dictionary with detection statistics
        """
        if not detections:
            return {
                'count': 0,
                'avg_confidence': 0.0,
                'avg_area': 0,
                'confidence_distribution': {}
            }
        
        confidences = [d['confidence'] for d in detections]
        areas = [d['area'] for d in detections]
        
        # Confidence distribution
        high_conf = sum(1 for c in confidences if c > 0.8)
        med_conf = sum(1 for c in confidences if 0.5 < c <= 0.8)
        low_conf = sum(1 for c in confidences if c <= 0.5)
        
        return {
            'count': len(detections),
            'avg_confidence': np.mean(confidences),
            'max_confidence': np.max(confidences),
            'min_confidence': np.min(confidences),
            'avg_area': np.mean(areas),
            'confidence_distribution': {
                'high (>0.8)': high_conf,
                'medium (0.5-0.8)': med_conf,
                'low (<0.5)': low_conf
            }
        }
    
    def update_config(self, new_config: Dict):
        """Update detector configuration"""
        self.config.update(new_config)
        
        # Update relevant parameters
        detection_config = self.config.get('detection', {})
        self.conf_threshold = detection_config.get('confidence_threshold', self.conf_threshold)
        
        basketball_config = self.config.get('basketball', {})
        self.min_player_area = basketball_config.get('min_player_area', self.min_player_area)
        self.max_player_area = basketball_config.get('max_player_area', self.max_player_area)
        self.min_aspect_ratio = basketball_config.get('min_aspect_ratio', self.min_aspect_ratio)
        self.max_aspect_ratio = basketball_config.get('max_aspect_ratio', self.max_aspect_ratio)
        
        print("YOLO Detector configuration updated")
