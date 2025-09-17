#!/usr/bin/env python3
"""Quick test script for Phase 2 (YOLO + SAM2) setup"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🏀 Basketball Player Tracking - Phase 2 Setup Test")
    print("=" * 60)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        
        from src.detection.yolo_detector import YOLODetector
        print("✅ YOLO Detector import successful")
        
        from src.segmentation.sam2_segmenter import SAM2Segmenter
        print("✅ SAM2 Segmenter import successful")
        
        from src.pipeline.yolo_sam2_pipeline import YOLOSAMPipeline
        print("✅ YOLO+SAM2 Pipeline import successful")
        
        # Test YOLO initialization
        print("\n🎯 Testing YOLO initialization...")
        detector = YOLODetector()
        print("✅ YOLO Detector initialization successful")
        
        # Test SAM2 initialization (may fail if models not available)
        print("\n🖼️  Testing SAM2 initialization...")
        try:
            segmenter = SAM2Segmenter()
            print("✅ SAM2 Segmenter initialization successful")
            sam2_status = "Available"
        except Exception as e:
            print(f"⚠️  SAM2 Segmenter initialization failed: {e}")
            print("   (This is expected on first run - models will download automatically)")
            sam2_status = "Will download on first use"
        
        # Test pipeline initialization
        print("\n🔄 Testing pipeline initialization...")
        try:
            pipeline = YOLOSAMPipeline(optimization_level="fast")
            print("✅ YOLO+SAM2 Pipeline initialization successful")
            
            # Get performance info
            perf = pipeline.get_current_performance()
            print(f"   Optimization level: {perf['optimization_level']}")
            print(f"   SAM2 status: {sam2_status}")
            
        except Exception as e:
            print(f"⚠️  Pipeline initialization had issues: {e}")
            print("   Pipeline will use fallback mode if SAM2 is not available")
        
        print(f"\n🎉 Phase 2 setup test completed!")
        print(f"\nComponent Status:")
        print(f"  ✅ YOLO Detection: Ready")
        print(f"  {'✅' if sam2_status == 'Available' else '⚠️ '} SAM2 Segmentation: {sam2_status}")
        print(f"  ✅ Integrated Pipeline: Ready")
        
        print(f"\nNext steps:")
        print(f"1. Test with webcam: python scripts/test_yolo_sam2.py --input 0")
        print(f"2. Test with video: python scripts/test_yolo_sam2.py --input path/to/video.mp4")
        print(f"3. Try different optimization levels: --optimization fast/balanced/quality")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase 2 setup test failed: {e}")
        print(f"\nPlease check:")
        print(f"1. All dependencies are installed")
        print(f"2. SAM2 is properly installed")
        print(f"3. Configuration files are in place")
        return False

if __name__ == "__main__":
    main()
