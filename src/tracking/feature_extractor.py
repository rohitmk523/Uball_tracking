"""Feature Extractor for Basketball Player Appearance

This module implements CNN-based feature extraction for basketball player
appearance matching in DeepSORT tracking. Optimized for basketball-specific
visual characteristics like jersey colors, player poses, and body shapes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import torchvision.transforms as transforms


class BasketballFeatureExtractor(nn.Module):
    """
    CNN feature extractor optimized for basketball players
    
    Extracts appearance features for player re-identification across frames.
    Designed to be robust to pose changes, lighting variations, and occlusions.
    """
    
    def __init__(self, feature_dim: int = 256, input_size: Tuple[int, int] = (128, 64)):
        """
        Initialize feature extractor
        
        Args:
            feature_dim: Dimension of output feature vector
            input_size: Input image size (height, width)
        """
        super(BasketballFeatureExtractor, self).__init__()
        
        self.feature_dim = feature_dim
        self.input_size = input_size
        
        # Convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            # First block: Basic edge and texture detection
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x32
            
            # Second block: Pattern recognition
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x16
            
            # Third block: Higher-level features
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x8
            
            # Fourth block: Complex pattern recognition
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 2))  # 4x2 = 8 spatial locations
        )
        
        # Calculate flattened feature size
        conv_output_size = 256 * 4 * 2  # 2048
        
        # Fully connected layers for final feature representation
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(conv_output_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor [batch_size, 3, height, width]
            
        Returns:
            Feature tensor [batch_size, feature_dim]
        """
        # Convolutional feature extraction
        x = self.conv_layers(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Final feature representation
        features = self.fc_layers(x)
        
        # L2 normalization for cosine similarity
        features = F.normalize(features, p=2, dim=1)
        
        return features


class PlayerAppearanceExtractor:
    """
    High-level interface for extracting player appearance features
    """
    
    def __init__(
        self, 
        feature_dim: int = 256,
        input_size: Tuple[int, int] = (128, 64),
        device: Optional[torch.device] = None
    ):
        """
        Initialize appearance extractor
        
        Args:
            feature_dim: Dimension of feature vectors
            input_size: Input image size (height, width)
            device: Computing device (auto-detected if None)
        """
        self.feature_dim = feature_dim
        self.input_size = input_size
        
        # Setup device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        # Initialize model
        self.model = BasketballFeatureExtractor(feature_dim, input_size)
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225]   # ImageNet stds
            )
        ])
        
        print(f"Player appearance extractor initialized on {self.device}")
        print(f"Feature dimension: {feature_dim}, Input size: {input_size}")
    
    def extract_features(
        self, 
        image: np.ndarray, 
        bboxes: List[List[int]], 
        masks: Optional[List[np.ndarray]] = None
    ) -> List[np.ndarray]:
        """
        Extract appearance features for multiple player detections
        
        Args:
            image: Full frame image (BGR format)
            bboxes: List of bounding boxes [[x1, y1, x2, y2], ...]
            masks: Optional list of segmentation masks
            
        Returns:
            List of feature vectors (numpy arrays)
        """
        if not bboxes:
            return []
        
        features = []
        batch_tensors = []
        
        # Preprocess all crops
        for i, bbox in enumerate(bboxes):
            crop = self._extract_player_crop(image, bbox, masks[i] if masks else None)
            if crop is not None:
                tensor = self._preprocess_crop(crop)
                batch_tensors.append(tensor)
            else:
                # Add zero feature for failed crops
                features.append(np.zeros(self.feature_dim, dtype=np.float32))
        
        # Batch inference if we have valid crops
        if batch_tensors:
            with torch.no_grad():
                batch_input = torch.stack(batch_tensors).to(self.device)
                batch_features = self.model(batch_input)
                batch_features = batch_features.cpu().numpy()
                
                # Add batch features to results
                batch_idx = 0
                for i, bbox in enumerate(bboxes):
                    if i < len(features):
                        continue  # Skip failed crops (already added zeros)
                    features.append(batch_features[batch_idx].astype(np.float32))
                    batch_idx += 1
        
        return features
    
    def _extract_player_crop(
        self, 
        image: np.ndarray, 
        bbox: List[int], 
        mask: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Extract and preprocess player crop from image
        
        Args:
            image: Full image
            bbox: Bounding box [x1, y1, x2, y2]
            mask: Optional segmentation mask
            
        Returns:
            Processed crop or None if invalid
        """
        x1, y1, x2, y2 = bbox
        
        # Validate bbox
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Ensure bbox is within image bounds
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        # Extract crop
        crop = image[y1:y2, x1:x2]
        
        if crop.size == 0:
            return None
        
        # Apply mask if available
        if mask is not None:
            mask_crop = mask[y1:y2, x1:x2]
            if mask_crop.size > 0:
                # Create 3-channel mask
                mask_3d = np.stack([mask_crop] * 3, axis=2).astype(np.uint8)
                crop = crop * mask_3d + (1 - mask_3d) * 128  # Gray background
        
        # Convert BGR to RGB
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        return crop
    
    def _preprocess_crop(self, crop: np.ndarray) -> torch.Tensor:
        """
        Preprocess crop for model input
        
        Args:
            crop: RGB crop image
            
        Returns:
            Preprocessed tensor
        """
        # Apply transformations
        tensor = self.transform(crop)
        return tensor
    
    def compute_distance_matrix(
        self, 
        features1: List[np.ndarray], 
        features2: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute distance matrix between two sets of features
        
        Args:
            features1: First set of features
            features2: Second set of features
            
        Returns:
            Distance matrix [len(features1), len(features2)]
        """
        if not features1 or not features2:
            return np.full((len(features1), len(features2)), 1.0)
        
        # Stack features
        f1 = np.stack(features1)  # [N1, feature_dim]
        f2 = np.stack(features2)  # [N2, feature_dim]
        
        # Compute cosine distances (1 - cosine_similarity)
        # Cosine similarity = dot(f1, f2) / (||f1|| * ||f2||)
        # Since features are L2-normalized, this simplifies to dot(f1, f2)
        similarities = np.dot(f1, f2.T)  # [N1, N2]
        distances = 1.0 - similarities
        
        # Ensure distances are in [0, 2] range and non-negative
        distances = np.clip(distances, 0.0, 2.0)
        
        return distances
    
    def get_feature_statistics(self, features: List[np.ndarray]) -> Dict:
        """
        Get statistics about extracted features
        
        Args:
            features: List of feature vectors
            
        Returns:
            Dictionary with feature statistics
        """
        if not features:
            return {'count': 0}
        
        feature_array = np.stack(features)
        
        return {
            'count': len(features),
            'feature_dim': feature_array.shape[1],
            'mean_norm': np.mean(np.linalg.norm(feature_array, axis=1)),
            'std_norm': np.std(np.linalg.norm(feature_array, axis=1)),
            'mean_feature': np.mean(feature_array, axis=0),
            'std_feature': np.std(feature_array, axis=0),
            'min_values': np.min(feature_array, axis=0),
            'max_values': np.max(feature_array, axis=0)
        }


class SimpleFeatureExtractor:
    """
    Simplified feature extractor using basic computer vision techniques
    
    Fallback option when deep learning features are not needed or available.
    Uses color histograms, texture features, and geometric properties.
    """
    
    def __init__(self, feature_dim: int = 256):
        """
        Initialize simple feature extractor
        
        Args:
            feature_dim: Target feature dimension
        """
        self.feature_dim = feature_dim
        
        # Feature component dimensions
        self.color_bins = 32  # Color histogram bins per channel
        self.texture_dim = 64  # LBP texture features
        self.geometric_dim = 16  # Geometric features
        
        print(f"Simple feature extractor initialized (dim={feature_dim})")
    
    def extract_features(
        self, 
        image: np.ndarray, 
        bboxes: List[List[int]], 
        masks: Optional[List[np.ndarray]] = None
    ) -> List[np.ndarray]:
        """Extract simple features for player crops"""
        features = []
        
        for i, bbox in enumerate(bboxes):
            crop = self._extract_crop(image, bbox, masks[i] if masks else None)
            if crop is not None:
                feature = self._extract_simple_features(crop)
                features.append(feature)
            else:
                features.append(np.zeros(self.feature_dim, dtype=np.float32))
        
        return features
    
    def _extract_crop(self, image: np.ndarray, bbox: List[int], mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Extract crop from image"""
        x1, y1, x2, y2 = bbox
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        crop = image[y1:y2, x1:x2]
        
        if mask is not None:
            mask_crop = mask[y1:y2, x1:x2]
            if mask_crop.size > 0:
                mask_3d = np.stack([mask_crop] * 3, axis=2).astype(np.uint8)
                crop = crop * mask_3d
        
        return crop
    
    def _extract_simple_features(self, crop: np.ndarray) -> np.ndarray:
        """Extract simple computer vision features"""
        features = []
        
        # Color histogram features (RGB)
        for channel in range(3):
            hist = cv2.calcHist([crop], [channel], None, [self.color_bins], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-7)  # Normalize
            features.extend(hist)
        
        # Geometric features
        h, w = crop.shape[:2]
        aspect_ratio = w / h if h > 0 else 1.0
        area = h * w
        
        geometric_features = [
            aspect_ratio,
            np.log(area + 1),
            h / 128.0,  # Normalized height
            w / 64.0,   # Normalized width
        ]
        
        # Pad or truncate to match feature_dim
        total_features = len(features) + len(geometric_features)
        if total_features < self.feature_dim:
            features.extend([0.0] * (self.feature_dim - total_features))
        else:
            features = features[:self.feature_dim - len(geometric_features)]
        
        features.extend(geometric_features)
        
        # L2 normalize
        feature_array = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(feature_array)
        if norm > 0:
            feature_array /= norm
        
        return feature_array
    
    def compute_distance_matrix(self, features1: List[np.ndarray], features2: List[np.ndarray]) -> np.ndarray:
        """Compute distance matrix using Euclidean distance"""
        if not features1 or not features2:
            return np.full((len(features1), len(features2)), 1.0)
        
        f1 = np.stack(features1)
        f2 = np.stack(features2)
        
        # Euclidean distances
        distances = np.sqrt(((f1[:, np.newaxis, :] - f2[np.newaxis, :, :]) ** 2).sum(axis=2))
        
        # Normalize to [0, 2] range
        max_dist = np.sqrt(2)  # Maximum distance between unit vectors
        distances = distances / max_dist
        
        return distances
