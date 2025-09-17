"""Segmentation module for precise player boundary detection using SAM2."""

try:
    from .sam2_segmenter import SAM2Segmenter
    __all__ = ["SAM2Segmenter"]
except ImportError as e:
    print(f"Warning: SAM2Segmenter import failed: {e}")
    __all__ = []
