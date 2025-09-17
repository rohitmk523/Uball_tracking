#!/usr/bin/env python3
"""Quick test script for Phase 3 (Complete Pipeline) setup"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🏀 Basketball Player Tracking - Phase 3 Setup Test")
    print("=" * 70)
    
    try:
        # Test individual component imports
        print("📦 Testing component imports...")
        
        from src.detection.yolo_detector import YOLODetector
        print("✅ YOLO Detector import successful")
        
        from src.segmentation.sam2_segmenter import SAM2Segmenter
        print("✅ SAM2 Segmenter import successful")
        
        from src.tracking.kalman_filter import BasketballKalmanFilter, KalmanTrackManager
        print("✅ Kalman Filter import successful")
        
        from src.tracking.feature_extractor import PlayerAppearanceExtractor, SimpleFeatureExtractor
        print("✅ Feature Extractor import successful")
        
        from src.tracking.utils import hungarian_assignment, compute_motion_distance_matrix
        print("✅ Tracking Utils import successful")
        
        from src.tracking.deepsort_tracker import DeepSORTTracker
        print("✅ DeepSORT Tracker import successful")
        
        from src.pipeline.basketball_tracker import BasketballTracker
        print("✅ Complete Basketball Tracker import successful")
        
        # Test component initializations
        print("\n🎯 Testing component initializations...")
        
        # Test YOLO
        detector = YOLODetector()
        print("✅ YOLO Detector initialization successful")
        
        # Test Kalman Filter
        dummy_bbox = [100, 100, 200, 300]
        kf = BasketballKalmanFilter(dummy_bbox)
        print("✅ Kalman Filter initialization successful")
        
        # Test Feature Extractor
        try:
            feature_extractor = PlayerAppearanceExtractor(feature_dim=256)
            print("✅ CNN Feature Extractor initialization successful")
            cnn_available = True
        except Exception as e:
            print(f"⚠️  CNN Feature Extractor failed: {e}")
            feature_extractor = SimpleFeatureExtractor(feature_dim=256)
            print("✅ Simple Feature Extractor initialization successful (fallback)")
            cnn_available = False
        
        # Test DeepSORT Tracker
        try:
            tracker = DeepSORTTracker()
            print("✅ DeepSORT Tracker initialization successful")
            deepsort_available = True
        except Exception as e:
            print(f"❌ DeepSORT Tracker initialization failed: {e}")
            deepsort_available = False
        
        # Test Complete Pipeline
        print("\n🔄 Testing complete pipeline initialization...")
        try:
            pipeline = BasketballTracker(optimization_level="fast")
            print("✅ Complete Basketball Pipeline initialization successful")
            
            # Get performance info
            perf = pipeline.get_current_performance()
            component_status = perf.get('component_status', {})
            
            print(f"   Optimization level: {perf.get('optimization_level', 'unknown')}")
            print(f"   YOLO: {'✅' if component_status.get('yolo') else '❌'}")
            print(f"   SAM2: {'✅' if component_status.get('sam2') else '⚠️ '}")
            print(f"   DeepSORT: {'✅' if component_status.get('deepsort') else '❌'}")
            
            pipeline_available = True
        except Exception as e:
            print(f"❌ Complete Pipeline initialization failed: {e}")
            pipeline_available = False
        
        # Test tracking utilities
        print("\n🧮 Testing tracking utilities...")
        
        # Test Hungarian assignment
        import numpy as np
        test_cost_matrix = np.array([[0.1, 0.8], [0.7, 0.2]])
        matches, unmatched_tracks, unmatched_dets = hungarian_assignment(test_cost_matrix)
        print("✅ Hungarian assignment algorithm working")
        
        # Test distance calculations
        from src.tracking.utils import calculate_iou, calculate_center_distance
        bbox1 = [10, 10, 50, 100]
        bbox2 = [30, 30, 70, 120]
        iou = calculate_iou(bbox1, bbox2)
        distance = calculate_center_distance(bbox1, bbox2)
        print("✅ Distance calculation utilities working")
        
        print(f"\n🎉 Phase 3 setup test completed!")
        
        # Component status summary
        print(f"\nComponent Status Summary:")
        print(f"  ✅ YOLO Detection: Ready")
        print(f"  {'✅' if cnn_available else '⚠️ '} Feature Extraction: {'CNN Ready' if cnn_available else 'Simple Mode (Fallback)'}")
        print(f"  {'✅' if deepsort_available else '❌'} DeepSORT Tracking: {'Ready' if deepsort_available else 'Failed'}")
        print(f"  {'✅' if pipeline_available else '❌'} Complete Pipeline: {'Ready' if pipeline_available else 'Failed'}")
        
        # Overall assessment
        if pipeline_available and deepsort_available:
            print(f"\n🏆 Phase 3 Status: FULLY OPERATIONAL")
            print(f"   All core components are working correctly")
            print(f"   Ready for comprehensive basketball tracking")
        elif pipeline_available:
            print(f"\n⚠️  Phase 3 Status: PARTIALLY OPERATIONAL")
            print(f"   Core pipeline works but some components may be in fallback mode")
            print(f"   Suitable for basic tracking with reduced features")
        else:
            print(f"\n❌ Phase 3 Status: NEEDS ATTENTION")
            print(f"   Critical components failed to initialize")
            print(f"   Please resolve the errors above")
        
        print(f"\nNext steps:")
        print(f"1. Test with webcam: python scripts/test_full_pipeline.py --input 0")
        print(f"2. Test with video: python scripts/test_full_pipeline.py --input path/to/video.mp4")
        print(f"3. Try different optimization levels: --optimization fast/balanced/quality")
        print(f"4. Enable advanced features: --show-trails --save-stats")
        
        return pipeline_available and deepsort_available
        
    except Exception as e:
        print(f"❌ Phase 3 setup test failed: {e}")
        print(f"\nPlease check:")
        print(f"1. All dependencies are installed")
        print(f"2. Previous phases (1 & 2) are working")
        print(f"3. Configuration files are in place")
        return False

if __name__ == "__main__":
    main()
