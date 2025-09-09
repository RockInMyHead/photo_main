#!/usr/bin/env python3
"""
Utility script to pre-download YOLO models.
This script downloads YOLO models to avoid delays during first-time usage.
"""

import sys
import os
from pathlib import Path

def download_yolo_models():
    """Download common YOLO models."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics not installed. Install with: pip install ultralytics")
        return False
    
    models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"]
    
    for model_name in models:
        print(f"Downloading {model_name}...")
        try:
            model = YOLO(model_name)
            print(f"✓ {model_name} downloaded successfully")
        except Exception as e:
            print(f"✗ Error downloading {model_name}: {e}")
    
    return True

if __name__ == "__main__":
    print("YOLO Model Downloader")
    print("=" * 50)
    success = download_yolo_models()
    if success:
        print("\nAll models downloaded. You can now use YOLO features without delays.")
    else:
        print("\nSome models failed to download. Check your internet connection and try again.")
        sys.exit(1)
