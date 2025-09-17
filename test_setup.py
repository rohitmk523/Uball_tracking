#!/usr/bin/env python3
"""Quick test script for basketball tracking setup"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("🏀 Basketball Player Tracking - Quick Setup Test")
    print("=" * 50)
    
    try:
        # Test YOLO detector import
        from src.detection.yolo_detector import YOLODetector
        print("✅ YOLO Detector import successful")
        
        # Test detector initialization
        detector = YOLODetector()
        print("✅ YOLO Detector initialization successful")
        
        print("\n🎉 Setup test completed successfully!")
        print("\nNext steps:")
        print("1. Run: python scripts/test_yolo.py --input 0")
        print("2. Or: python scripts/test_yolo.py --input path/to/video.mp4")
        
    except Exception as e:
        print(f"❌ Setup test failed: {e}")
        print("\nPlease check:")
        print("1. All dependencies are installed: poetry install")
        print("2. Models are downloaded properly")
        print("3. Configuration files are in place")

if __name__ == "__main__":
    main()
