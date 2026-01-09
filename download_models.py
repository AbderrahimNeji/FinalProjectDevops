#!/usr/bin/env python3
"""
Download YOLOv8 models for road degradation detection.
This script downloads pre-trained YOLOv8 models to the models/ directory.
"""

import os
import sys
import shutil
from pathlib import Path

def download_models():
    """Download YOLOv8 models to models/ directory"""
    try:
        from ultralytics import YOLO
        
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        # Models to download
        model_names = [
            'yolov8n.pt',  # Nano - fastest
            'yolov8m.pt',  # Medium - balanced (recommended)
            'yolov8l.pt',  # Large - more accurate
        ]
        
        print("🚀 Downloading YOLOv8 models to models/ directory...")
        print()
        
        for model_name in model_names:
            model_path = models_dir / model_name
            
            # Skip if already exists
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                print(f"✅ {model_name} already exists ({size_mb:.1f} MB)")
                continue
            
            print(f"⏳ Downloading {model_name}...")
            try:
                # Download model (saves to cache)
                model = YOLO(model_name)
                
                # Copy to models/ directory
                cache_path = Path(model.ckpt_path)
                shutil.copy2(cache_path, model_path)
                
                size_mb = model_path.stat().st_size / (1024 * 1024)
                print(f"✅ Downloaded and copied {model_name} to models/ ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"❌ Failed to download {model_name}: {e}")
        
        print()
        print("✅ All models downloaded successfully!")
        print(f"📁 Location: {models_dir.absolute()}")
        
        # List downloaded files
        print()
        print("📋 Downloaded files:")
        for file in sorted(models_dir.glob('*.pt')):
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  • {file.name} ({size_mb:.1f} MB)")
    
    except ImportError:
        print("❌ YOLOv8 not installed!")
        print("Install it with: pip install ultralytics")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    download_models()
