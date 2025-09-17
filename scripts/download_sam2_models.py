#!/usr/bin/env python3
"""Download SAM2 Models

This script downloads the required SAM2 models for basketball player segmentation.
"""

import os
import sys
import urllib.request
from pathlib import Path
import hashlib

# SAM2 model URLs and checksums
SAM2_MODELS = {
    "sam2_hiera_tiny.pt": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
        "size": "38.9 MB"
    },
    "sam2_hiera_small.pt": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt", 
        "size": "185 MB"
    },
    "sam2_hiera_base_plus.pt": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
        "size": "319 MB"
    },
    "sam2_hiera_large.pt": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
        "size": "899 MB"
    }
}

def download_file(url: str, filepath: str, description: str = ""):
    """Download file with progress bar"""
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded * 100.0 / total_size
            sys.stdout.write(f"\r{description}: {percent:.1f}% ({downloaded//1024//1024} MB / {total_size//1024//1024} MB)")
            sys.stdout.flush()
    
    try:
        print(f"Downloading {description}...")
        urllib.request.urlretrieve(url, filepath, progress_hook)
        print(f"\n✅ {description} downloaded successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Failed to download {description}: {e}")
        return False

def main():
    print("🏀 SAM2 Model Downloader for Basketball Tracking")
    print("=" * 60)
    
    # Create models directory
    models_dir = Path("models/sam2")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Models will be saved to: {models_dir}")
    
    # Ask user which models to download
    print("\nAvailable SAM2 models:")
    print("1. sam2_hiera_tiny.pt (38.9 MB) - Fastest, good for real-time")
    print("2. sam2_hiera_small.pt (185 MB) - Balanced speed/quality") 
    print("3. sam2_hiera_base_plus.pt (319 MB) - Good quality")
    print("4. sam2_hiera_large.pt (899 MB) - Best quality (default)")
    print("5. Download all models")
    
    choice = input("\nWhich model(s) would you like to download? (1-5, default=4): ").strip()
    
    if not choice:
        choice = "4"
    
    models_to_download = []
    
    if choice == "1":
        models_to_download = ["sam2_hiera_tiny.pt"]
    elif choice == "2": 
        models_to_download = ["sam2_hiera_small.pt"]
    elif choice == "3":
        models_to_download = ["sam2_hiera_base_plus.pt"] 
    elif choice == "4":
        models_to_download = ["sam2_hiera_large.pt"]
    elif choice == "5":
        models_to_download = list(SAM2_MODELS.keys())
    else:
        print("Invalid choice, downloading large model by default...")
        models_to_download = ["sam2_hiera_large.pt"]
    
    # Download selected models
    success_count = 0
    for model_name in models_to_download:
        model_path = models_dir / model_name
        
        # Skip if already exists
        if model_path.exists():
            print(f"✅ {model_name} already exists, skipping...")
            success_count += 1
            continue
        
        model_info = SAM2_MODELS[model_name]
        success = download_file(
            model_info["url"], 
            str(model_path),
            f"{model_name} ({model_info['size']})"
        )
        
        if success:
            success_count += 1
    
    print(f"\n🎉 Download complete! {success_count}/{len(models_to_download)} models downloaded successfully.")
    
    # Show next steps
    print("\n📋 Next steps:")
    print("1. Test SAM2 segmentation:")
    print("   python scripts/test_yolo_sam2.py --input test_videos/sample60s_video-2.mp4")
    print("2. Test complete pipeline:")
    print("   python scripts/test_full_pipeline.py --input test_videos/sample60s_video-2.mp4")
    
    return success_count == len(models_to_download)

if __name__ == "__main__":
    main()

