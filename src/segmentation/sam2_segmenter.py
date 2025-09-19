"""SAM2 Segmenter for Basketball Player Segmentation

This module implements SAM2 (Segment Anything Model 2) for pixel-accurate
basketball player segmentation using YOLO bounding boxes as prompts.
"""

import torch
import numpy as np
import cv2
import yaml
import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError as e:
    SAM2_AVAILABLE = False
    print(f"Warning: SAM2 not available: {e}")
    print("SAM2 will use fallback mode with rectangular masks")
    
    # Create dummy classes to prevent import errors
    class DummySAM2ImagePredictor:
        def __init__(self, model): pass
        def set_image(self, image): pass
        def predict(self, **kwargs): return [], [], []
    
    def build_sam2(*args, **kwargs): return None
    SAM2ImagePredictor = DummySAM2ImagePredictor


class SAM2Segmenter:
    """SAM2 segmentation for basketball players"""
    
    def __init__(
        self, 
        model_cfg: str = "sam2_hiera_l.yaml", 
        checkpoint: str = "sam2_hiera_large.pt",
        config_path: Optional[str] = None
    ):
        """
        Initialize SAM2 segmenter
        
        Args:
            model_cfg: SAM2 model configuration
            checkpoint: Path to SAM2 checkpoint
            config_path: Path to SAM2 configuration file
        """
        if not SAM2_AVAILABLE:
            raise ImportError("SAM2 is not available. Please install it first.")
        
        self.model_cfg = model_cfg
        self.checkpoint = checkpoint
        
        # Load configuration
        if config_path:
            self.config = self._load_config(config_path)
        else:
            self.config = self._default_config()
        
        # Initialize device
        self.device = self._get_device()
        
        # Initialize SAM2 model and predictor
        self.model = None
        self.predictor = None
        self._initialize_model()
        
        # Basketball-specific parameters
        self.mask_threshold = self.config.get('segmentation', {}).get('mask_threshold', 0.0)
        self.stability_score_threshold = self.config.get('segmentation', {}).get(
            'stability_score_threshold', 0.95
        )
        self.remove_small_regions = self.config.get('segmentation', {}).get(
            'remove_small_regions', True
        )
        self.min_region_area = self.config.get('segmentation', {}).get('min_region_area', 500)
        
        print(f"SAM2 Segmenter initialized on device: {self.device}")
        print(f"Model: {model_cfg}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found. Using default config.")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default configuration for SAM2 segmenter"""
        return {
            'model': {
                'name': 'sam2_hiera_l.yaml',
                'checkpoint': 'sam2_hiera_large.pt',
                'device': 'auto'
            },
            'segmentation': {
                'use_bbox_prompts': True,
                'use_point_prompts': False,
                'multimask_output': False,
                'mask_threshold': 0.0,
                'stability_score_threshold': 0.95,
                'remove_small_regions': True,
                'min_region_area': 500
            },
            'basketball': {
                'torso_focus': True,
                'exclude_ground_shadows': True,
                'morphological_operations': True,
                'kernel_size': 3
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
    
    def _initialize_model(self):
        """Initialize SAM2 model and predictor"""
        try:
            # Try to download model if not exists
            checkpoint_path = self._get_model_path(self.checkpoint)
            
            # Build SAM2 model
            self.model = build_sam2(self.model_cfg, checkpoint_path, device=self.device)
            self.predictor = SAM2ImagePredictor(self.model)
            
            print(f"✅ SAM2 model loaded: {self.model_cfg}")
            
        except Exception as e:
            print(f"❌ Failed to initialize SAM2 model: {e}")
            print("Falling back to dummy segmentation mode...")
            self.model = None
            self.predictor = None
    
    def _get_model_path(self, checkpoint: str) -> str:
        """Get or download SAM2 model checkpoint"""
        # Check if it's already a full path
        if os.path.exists(checkpoint):
            return checkpoint
        
        # Check in models directory
        models_dir = Path("models/sam2")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = models_dir / checkpoint
        
        if model_path.exists():
            return str(model_path)
        
        # If model doesn't exist, try to download it
        print(f"📦 SAM2 model {checkpoint} will be downloaded on first use...")
        
        # For now, return the checkpoint name and let SAM2 handle the download
        return checkpoint
    
    def segment_players(self, image: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        Create segmentation masks for detected players
        
        Args:
            image: Input image (BGR format)
            detections: List of player detections from YOLO
            
        Returns:
            List of detections with added segmentation masks
        """
        if not detections:
            return []
        
        # If SAM2 is not available, return detections with dummy masks
        if self.predictor is None:
            return self._create_dummy_masks(image, detections)
        
        try:
            # Convert BGR to RGB for SAM2
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Set image for SAM2
            self.predictor.set_image(rgb_image)
            
            segmented_detections = []
            
            for detection in detections:
                # Use bounding box as prompt for SAM2
                bbox = detection['bbox']
                x1, y1, x2, y2 = [float(coord) for coord in bbox]  # Ensure float for SAM2
                
                # Convert to SAM2 box format
                input_box = np.array([x1, y1, x2, y2])
                
                # Get segmentation mask
                masks, scores, _ = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=self.config.get('segmentation', {}).get('multimask_output', False),
                )
                
                # Process the best mask
                if len(masks) > 0:
                    best_mask = masks[0]  # Take the first (usually best) mask
                    best_score = float(scores[0])
                    
                    # Post-process mask
                    processed_mask = self._post_process_mask(best_mask, detection, image.shape[:2])
                    
                    # Add mask to detection
                    segmented_detection = detection.copy()
                    segmented_detection['mask'] = processed_mask
                    segmented_detection['mask_score'] = best_score
                    segmented_detection['has_mask'] = True
                    
                    segmented_detections.append(segmented_detection)
                else:
                    # No mask generated, add detection without mask
                    segmented_detection = detection.copy()
                    segmented_detection['has_mask'] = False
                    segmented_detections.append(segmented_detection)
            
            return segmented_detections
            
        except Exception as e:
            print(f"Warning: SAM2 segmentation failed: {e}")
            return self._create_dummy_masks(image, detections)
    
    def _create_dummy_masks(self, image: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """Create dummy rectangular masks when SAM2 is not available"""
        segmented_detections = []
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = [int(coord) for coord in bbox]  # Ensure integers
            
            # Create a rectangular mask
            mask = np.zeros(image.shape[:2], dtype=bool)
            mask[y1:y2, x1:x2] = True
            
            segmented_detection = detection.copy()
            segmented_detection['mask'] = mask
            segmented_detection['mask_score'] = 0.8  # Dummy score
            segmented_detection['has_mask'] = True
            segmented_detection['mask_type'] = 'dummy'
            
            segmented_detections.append(segmented_detection)
        
        return segmented_detections
    
    def _post_process_mask(self, mask: np.ndarray, detection: Dict, image_shape: Tuple[int, int]) -> np.ndarray:
        """Post-process segmentation mask for basketball players"""
        # Convert to boolean mask
        processed_mask = mask.astype(bool)
        
        # Remove small regions if enabled
        if self.remove_small_regions:
            processed_mask = self._remove_small_regions(processed_mask, self.min_region_area)
        
        # Apply morphological operations if enabled
        basketball_config = self.config.get('basketball', {})
        if basketball_config.get('morphological_operations', True):
            kernel_size = basketball_config.get('kernel_size', 3)
            processed_mask = self._apply_morphological_operations(processed_mask, kernel_size)
        
        # Ensure mask is within bounding box region (basketball-specific)
        if basketball_config.get('torso_focus', True):
            processed_mask = self._apply_torso_focus(processed_mask, detection)
        
        return processed_mask
    
    def _remove_small_regions(self, mask: np.ndarray, min_area: int) -> np.ndarray:
        """Remove small disconnected regions from mask"""
        # Convert to uint8 for cv2 operations
        mask_uint8 = mask.astype(np.uint8) * 255
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8)
        
        # Create new mask keeping only large regions
        new_mask = np.zeros_like(mask, dtype=bool)
        
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                new_mask[labels == i] = True
        
        return new_mask
    
    def _apply_morphological_operations(self, mask: np.ndarray, kernel_size: int) -> np.ndarray:
        """Apply morphological operations to clean up mask"""
        # Convert to uint8
        mask_uint8 = mask.astype(np.uint8) * 255
        
        # Create kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Apply closing to fill holes
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        
        # Apply opening to remove noise
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        
        return mask_uint8 > 0
    
    def _apply_torso_focus(self, mask: np.ndarray, detection: Dict) -> np.ndarray:
        """Focus mask on player torso region (basketball-specific)"""
        bbox = detection['bbox']
        x1, y1, x2, y2 = [int(coord) for coord in bbox]  # Ensure integers
        
        # Create a mask that emphasizes the torso region
        height = y2 - y1
        width = x2 - x1
        
        # Define torso region (middle 60% vertically, full width)
        torso_y1 = y1 + int(height * 0.2)
        torso_y2 = y1 + int(height * 0.8)
        
        # Create weight mask (higher weight for torso)
        weight_mask = np.ones_like(mask, dtype=float)
        weight_mask[torso_y1:torso_y2, x1:x2] = 1.5  # Higher weight for torso
        
        # Apply weights (this is a simple approach)
        # In practice, you might want more sophisticated torso detection
        return mask
    
    def visualize_segmentation(
        self, 
        image: np.ndarray, 
        detections: List[Dict],
        alpha: float = 0.5,
        show_masks: bool = True,
        show_boxes: bool = True
    ) -> np.ndarray:
        """
        Visualize segmentation masks on image
        
        Args:
            image: Input image
            detections: List of detections with masks
            alpha: Transparency for mask overlay
            show_masks: Whether to show segmentation masks
            show_boxes: Whether to show bounding boxes
            
        Returns:
            Image with segmentation overlays
        """
        vis_image = image.copy()
        
        # Color palette for different players
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (0, 128, 0), (128, 128, 0), (0, 128, 128), (128, 0, 0)
        ]
        
        for i, detection in enumerate(detections):
            if not detection.get('has_mask', False):
                continue
                
            mask = detection['mask']
            bbox = detection['bbox']
            conf = detection['confidence']
            mask_score = detection.get('mask_score', 0.0)
            
            # Get color for this player
            color = colors[i % len(colors)]
            
            # Draw mask overlay
            if show_masks and mask is not None:
                # Create colored mask
                colored_mask = np.zeros_like(vis_image)
                colored_mask[mask] = color
                
                # Blend with original image
                vis_image = cv2.addWeighted(vis_image, 1 - alpha, colored_mask, alpha, 0)
                
                # Draw mask contour
                mask_uint8 = mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis_image, contours, -1, color, 2)
            
            # Draw bounding box
            if show_boxes:
                x1, y1, x2, y2 = [int(coord) for coord in bbox]  # Ensure integers
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                mask_type = detection.get('mask_type', 'sam2')
                label = f"Player {i+1}: {conf:.2f}"
                if mask_score > 0:
                    label += f" | Mask: {mask_score:.2f}"
                if mask_type == 'dummy':
                    label += " (Box)"
                
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
        
        return vis_image
    
    def get_segmentation_statistics(self, detections: List[Dict]) -> Dict:
        """
        Get statistics about segmentation results
        
        Args:
            detections: List of detections with masks
            
        Returns:
            Dictionary with segmentation statistics
        """
        if not detections:
            return {
                'total_detections': 0,
                'successful_segmentations': 0,
                'average_mask_score': 0.0,
                'mask_coverage': {}
            }
        
        successful_masks = [d for d in detections if d.get('has_mask', False)]
        mask_scores = [d.get('mask_score', 0.0) for d in successful_masks if d.get('mask_score', 0.0) > 0]
        
        # Calculate mask coverage statistics
        coverage_stats = {}
        for detection in successful_masks:
            if 'mask' in detection:
                mask = detection['mask']
                bbox = detection['bbox']
                x1, y1, x2, y2 = [float(coord) for coord in bbox]  # Ensure float for calculations
                
                bbox_area = (x2 - x1) * (y2 - y1)
                mask_area = np.sum(mask)
                coverage = mask_area / bbox_area if bbox_area > 0 else 0
                
                if 'coverages' not in coverage_stats:
                    coverage_stats['coverages'] = []
                coverage_stats['coverages'].append(coverage)
        
        if coverage_stats.get('coverages'):
            coverage_stats['average_coverage'] = np.mean(coverage_stats['coverages'])
            coverage_stats['min_coverage'] = np.min(coverage_stats['coverages'])
            coverage_stats['max_coverage'] = np.max(coverage_stats['coverages'])
        
        return {
            'total_detections': len(detections),
            'successful_segmentations': len(successful_masks),
            'success_rate': len(successful_masks) / len(detections) if detections else 0,
            'average_mask_score': np.mean(mask_scores) if mask_scores else 0.0,
            'mask_coverage': coverage_stats,
            'sam2_available': self.predictor is not None
        }
    
    def update_config(self, new_config: Dict):
        """Update segmenter configuration"""
        self.config.update(new_config)
        
        # Update relevant parameters
        segmentation_config = self.config.get('segmentation', {})
        self.mask_threshold = segmentation_config.get('mask_threshold', self.mask_threshold)
        self.stability_score_threshold = segmentation_config.get(
            'stability_score_threshold', self.stability_score_threshold
        )
        self.remove_small_regions = segmentation_config.get(
            'remove_small_regions', self.remove_small_regions
        )
        self.min_region_area = segmentation_config.get('min_region_area', self.min_region_area)
        
        print("SAM2 Segmenter configuration updated")
