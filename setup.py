#!/usr/bin/env python3
"""
Setup script for AI Workout Form Corrector
Handles installation and initial configuration
"""

import subprocess
import sys
import os
import pkg_resources

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    else:
        print(f"✅ Python version {sys.version.split()[0]} is compatible")

def install_requirements():
    """Install required packages"""
    try:
        print("📦 Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        sys.exit(1)

def verify_camera_access():
    """Test camera access"""
    try:
        import cv2
        print("📹 Testing camera access...")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera access confirmed")
            cap.release()
        else:
            print("⚠️  Camera not accessible - check permissions")
    except ImportError:
        print("⚠️  OpenCV not installed - run installation first")

def create_directories():
    """Create necessary directories"""
    dirs = ['models', 'data', 'exports']
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"📁 Created directory: {dir_name}")

def main():
    """Main setup function"""
    print("🚀 AI Workout Form Corrector Setup")
    print("=" * 40)

    check_python_version()
    install_requirements()
    create_directories()
    verify_camera_access()

    print("\n🎉 Setup completed successfully!")
    print("\nTo run the application:")
    print("  Basic version: python ai_workout_form_corrector.py")
    print("  ML version:    python ai_workout_form_corrector_ml.py")

if __name__ == "__main__":
    main()
