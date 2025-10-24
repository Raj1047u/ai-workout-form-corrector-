#!/usr/bin/env python3
"""
Test script for AI Workout Form Corrector
Verifies that all components are working correctly
"""

import sys
import cv2
import numpy as np

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing package imports...")

    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False

    try:
        import mediapipe as mp
        print("✅ MediaPipe imported successfully")
    except ImportError as e:
        print(f"❌ MediaPipe import failed: {e}")
        return False

    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False

    try:
        import tkinter as tk
        print("✅ Tkinter imported successfully")
    except ImportError as e:
        print(f"❌ Tkinter import failed: {e}")
        return False

    return True

def test_camera():
    """Test camera functionality"""
    print("\n📹 Testing camera functionality...")

    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return False

        ret, frame = cap.read()
        if not ret:
            print("❌ Cannot read from camera")
            cap.release()
            return False

        print(f"✅ Camera working - Frame size: {frame.shape}")
        cap.release()
        return True

    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_mediapipe():
    """Test MediaPipe pose estimation"""
    print("\n🤖 Testing MediaPipe pose estimation...")

    try:
        import mediapipe as mp

        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Create a dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = pose.process(dummy_image)

        print("✅ MediaPipe pose estimation initialized")
        pose.close()
        return True

    except Exception as e:
        print(f"❌ MediaPipe test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 AI Workout Form Corrector - Component Tests")
    print("=" * 50)

    tests = [
        ("Package Imports", test_imports),
        ("Camera Access", test_camera),
        ("MediaPipe Pose", test_mediapipe)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! System is ready.")
        return 0
    else:
        print("❌ Some tests failed. Check the setup.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
