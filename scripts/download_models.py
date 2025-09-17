#!/usr/bin/env python3
"""Download and Setup Models for Basketball Player Tracking

This script downloads all required model weights and sets up the project
for basketball player tracking.
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path
import argparse


def create_directories():
    """Create necessary model directories"""
    directories = [
        'models/yolo',
        'models/sam2', 
        'models/deepsort',
        'data/sample_videos',
        'data/test_images',
        'outputs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")


def download_yolo_models():
    """Download YOLO models (they will be downloaded automatically on first use)"""
    print("\n🎯 YOLO Models Setup")
    print("-" * 30)
    
    # YOLO models are downloaded automatically by ultralytics
    # We just need to create a test to trigger the download
    try:
        from ultralytics import YOLO
        
        print("📦 Testing YOLO11 Nano model download...")
        model = YOLO('yolo11n.pt')  # This will download the model
        print("✅ YOLO11 Nano model ready")
        
        # Also prepare other YOLO variants
        variants = ['yolo11s.pt', 'yolo11m.pt']
        for variant in variants:
            try:
                print(f"📦 Preparing {variant}...")
                YOLO(variant)
                print(f"✅ {variant} ready")
            except Exception as e:
                print(f"⚠️  {variant} not downloaded: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to setup YOLO models: {e}")
        return False


def setup_sam2():
    """Setup SAM2 (will be downloaded on first use)"""
    print("\n🖼️  SAM2 Models Setup")
    print("-" * 30)
    
    try:
        # SAM2 models will be downloaded automatically when first used
        print("📦 SAM2 models will be downloaded automatically on first use")
        
        # Create placeholder files
        sam2_dir = Path("models/sam2")
        readme_content = """# SAM2 Models

SAM2 models will be downloaded automatically when first used.

Available models:
- sam2_hiera_tiny.pt (fastest)
- sam2_hiera_small.pt (balanced)
- sam2_hiera_base_plus.pt (good quality)
- sam2_hiera_large.pt (best quality)

The system will automatically choose the appropriate model based on your hardware.
"""
        
        with open(sam2_dir / "README.md", "w") as f:
            f.write(readme_content)
        
        print("✅ SAM2 setup complete")
        return True
        
    except Exception as e:
        print(f"❌ Failed to setup SAM2: {e}")
        return False


def setup_deepsort():
    """Setup DeepSORT models"""
    print("\n👥 DeepSORT Models Setup")
    print("-" * 30)
    
    try:
        # DeepSORT uses a simple CNN that we define in code
        # No external models needed for basic version
        
        deepsort_dir = Path("models/deepsort")
        readme_content = """# DeepSORT Models

DeepSORT uses a built-in CNN feature extractor for basic tracking.

For advanced tracking, you can add pre-trained models:
- OSNet models for better appearance features
- Custom trained models for basketball-specific features

The basic implementation uses a simple CNN defined in the code.
"""
        
        with open(deepsort_dir / "README.md", "w") as f:
            f.write(readme_content)
        
        print("✅ DeepSORT setup complete")
        return True
        
    except Exception as e:
        print(f"❌ Failed to setup DeepSORT: {e}")
        return False


def download_sample_data():
    """Download or create sample data for testing"""
    print("\n📹 Sample Data Setup")
    print("-" * 30)
    
    try:
        sample_dir = Path("data/sample_videos")
        
        # Create a README with instructions for sample data
        readme_content = """# Sample Videos

Place your basketball video files here for testing.

Recommended video characteristics:
- Format: MP4, AVI, MOV
- Resolution: 720p or higher
- Frame rate: 30 FPS
- Duration: 30 seconds to 5 minutes for testing
- Camera angle: Elevated (sideline or corner view)
- Multiple players visible
- Good lighting conditions

For testing without sample videos:
- Use webcam (camera index 0)
- Use any basketball video you have

Example command:
```bash
python scripts/test_yolo.py --input data/sample_videos/your_video.mp4
```
"""
        
        with open(sample_dir / "README.md", "w") as f:
            f.write(readme_content)
        
        # Create sample images directory info
        images_dir = Path("data/test_images")
        images_readme = """# Test Images

Place basketball images here for testing detection on static images.

Recommended image characteristics:
- Format: JPG, PNG
- Resolution: 640x480 or higher
- Multiple players visible
- Clear player visibility
- Basketball court setting
"""
        
        with open(images_dir / "README.md", "w") as f:
            f.write(images_readme)
        
        print("✅ Sample data directories set up")
        return True
        
    except Exception as e:
        print(f"❌ Failed to setup sample data: {e}")
        return False


def verify_dependencies():
    """Verify all required dependencies are installed"""
    print("\n🔍 Verifying Dependencies")
    print("-" * 30)
    
    required_packages = [
        'torch',
        'torchvision', 
        'ultralytics',
        'opencv-python',
        'numpy',
        'pillow',
        'matplotlib',
        'filterpy',
        'scipy',
        'lap',
        'easydict',
        'yaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
                print(f"✅ {package} (OpenCV {cv2.__version__})")
            elif package == 'yaml':
                import yaml
                print(f"✅ PyYAML")
            else:
                __import__(package)
                print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - MISSING")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using:")
        print("poetry install")
        return False
    else:
        print("\n✅ All dependencies verified!")
        return True


def test_gpu_availability():
    """Test GPU availability for acceleration"""
    print("\n🚀 GPU Availability Test")
    print("-" * 30)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available: {gpu_count} GPU(s)")
            print(f"   Primary GPU: {gpu_name}")
            
            # Test CUDA memory
            device = torch.device('cuda:0')
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**2
            print(f"   GPU Memory: {memory_allocated:.1f}MB allocated, {memory_reserved:.1f}MB reserved")
            
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✅ Apple Metal Performance Shaders (MPS) available")
            print("   Using Apple Silicon GPU acceleration")
            
        else:
            print("⚠️  No GPU acceleration available - using CPU")
            print("   Consider using a machine with NVIDIA GPU or Apple Silicon for better performance")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test GPU availability: {e}")
        return False


def create_quick_test_script():
    """Create a quick test script for immediate validation"""
    test_script = """#!/usr/bin/env python3
\"\"\"Quick test script for basketball tracking setup\"\"\"

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
        
        print("\\n🎉 Setup test completed successfully!")
        print("\\nNext steps:")
        print("1. Run: python scripts/test_yolo.py --input 0")
        print("2. Or: python scripts/test_yolo.py --input path/to/video.mp4")
        
    except Exception as e:
        print(f"❌ Setup test failed: {e}")
        print("\\nPlease check:")
        print("1. All dependencies are installed: poetry install")
        print("2. Models are downloaded properly")
        print("3. Configuration files are in place")

if __name__ == "__main__":
    main()
"""
    
    with open("test_setup.py", "w") as f:
        f.write(test_script)
    
    # Make it executable
    os.chmod("test_setup.py", 0o755)
    print("✅ Created quick test script: test_setup.py")


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description='Download and setup basketball tracking models')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip model downloads (setup directories only)')
    parser.add_argument('--gpu-test', action='store_true',
                       help='Run GPU availability test')
    
    args = parser.parse_args()
    
    print("🏀 Basketball Player Tracking System Setup")
    print("=" * 50)
    
    success = True
    
    # Create directories
    print("\n1. Creating project directories...")
    create_directories()
    
    # Verify dependencies
    print("\n2. Verifying dependencies...")
    if not verify_dependencies():
        success = False
    
    # Test GPU if requested
    if args.gpu_test:
        print("\n3. Testing GPU availability...")
        test_gpu_availability()
    
    if not args.skip_download:
        # Download models
        print("\n4. Setting up models...")
        if not download_yolo_models():
            success = False
        
        if not setup_sam2():
            success = False
        
        if not setup_deepsort():
            success = False
        
        # Setup sample data
        print("\n5. Setting up sample data...")
        if not download_sample_data():
            success = False
    
    # Create quick test script
    print("\n6. Creating test scripts...")
    create_quick_test_script()
    
    # Final status
    print("\n" + "=" * 50)
    if success:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Test the setup: python test_setup.py")
        print("2. Run YOLO test: python scripts/test_yolo.py --input 0")
        print("3. Check the implementation guide for next phases")
        print("\nFor help: python scripts/test_yolo.py --help")
    else:
        print("❌ Setup completed with errors")
        print("Please resolve the issues above and run setup again")
    
    return success


if __name__ == "__main__":
    main()
