"""Pipeline module for complete basketball tracking workflow."""

# Import available pipeline components
try:
    from .yolo_sam2_pipeline import YOLOSAMPipeline
    _available_imports = ["YOLOSAMPipeline"]
except ImportError:
    _available_imports = []

# These will be available in later phases
try:
    from .basketball_tracker import BasketballTracker
    _available_imports.append("BasketballTracker")
except ImportError:
    pass

try:
    from .team_classifier import TeamClassifier
    _available_imports.append("TeamClassifier")
except ImportError:
    pass

__all__ = _available_imports
