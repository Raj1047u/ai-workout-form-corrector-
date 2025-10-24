# Create additional utility files and documentation

import os

# Create a comprehensive README file
readme_content = '''# AI Workout Form Corrector

A comprehensive real-time exercise form analysis system using computer vision and machine learning, built according to the academic research specifications from Chandigarh Engineering College.

## 🎯 Project Overview

The AI Workout Form Corrector is a desktop application that leverages MediaPipe Pose estimation and biomechanical analysis to provide real-time feedback on exercise form. The system analyzes three foundational exercises:

- **Squats**: Depth analysis, back posture, knee alignment
- **Push-ups**: Body alignment, range of motion, core engagement
- **Lunges**: Knee angles, front knee position, balance

## 🚀 Key Features

### Core Functionality
- **Real-time Pose Estimation**: Uses Google's MediaPipe for 33-point body landmark detection
- **Biomechanical Analysis**: Rule-based engine grounded in exercise science principles
- **Form-Gated Repetition Counting**: Only counts correctly performed repetitions
- **Real-time Feedback**: Immediate corrective instructions and encouragement
- **Performance Monitoring**: Achieves 24-28 FPS on consumer hardware

### Enhanced ML Version Features
- **Machine Learning Models**: Random Forest classifiers for each exercise
- **Hybrid Analysis**: Combines rule-based and ML approaches
- **Progress Analytics**: Workout history and performance tracking
- **Data Recording**: Capture training data for model improvement
- **Advanced Metrics**: Confidence scores, form quality assessment

## 📋 Requirements

### Hardware Requirements
- Standard webcam (built-in or USB)
- Consumer-grade computer (Intel Core i5 or equivalent)
- 8GB RAM minimum
- No dedicated GPU required

### Software Requirements
- Python 3.9 or higher
- See `requirements.txt` for detailed dependencies

## 🛠️ Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the basic version**:
   ```bash
   python ai_workout_form_corrector.py
   ```

4. **Run the ML-enhanced version**:
   ```bash
   python ai_workout_form_corrector_ml.py
   ```

## 📖 Usage Guide

### Getting Started
1. Launch the application
2. Position yourself 3-6 feet from the camera
3. Select your desired exercise (Squat, Push-up, or Lunge)
4. Click "START WORKOUT" to begin
5. Follow the real-time feedback displayed on screen

### Camera Positioning
- **Squats & Lunges**: Position camera to the side for profile view
- **Push-ups**: Position camera to capture full body from side or slight angle
- Ensure adequate lighting and minimal background clutter
- Keep entire body visible within the camera frame

### Understanding Feedback
- **Green messages**: Encouragement and successful repetitions
- **Yellow/Red messages**: Form corrections needed
- **Rep counter**: Only counts properly executed repetitions
- **Stage indicator**: Shows current phase of exercise (up/down)

## 🧠 Technical Architecture

### Core Components

1. **Pose Estimation Module**
   - MediaPipe Pose framework
   - 33 anatomical landmarks
   - Real-time 2D coordinate extraction

2. **Form Analysis Engine**
   - Geometric angle calculations
   - Biomechanical rule evaluation
   - Exercise-specific state machines

3. **Feedback System**
   - Real-time visual overlays
   - Text-based corrective instructions
   - Form-gated repetition counting

4. **User Interface**
   - Tkinter-based GUI
   - Video feed integration
   - Control panels and statistics

### ML Enhancement Features

1. **Machine Learning Models**
   - Random Forest classifiers per exercise
   - Feature extraction from pose data
   - Form quality prediction

2. **Analytics Dashboard**
   - Progress tracking over time
   - Performance metrics visualization
   - Data export capabilities

3. **Training Interface**
   - Data recording functionality
   - Model retraining capabilities
   - Custom dataset integration

## 📊 Performance Metrics

Based on academic evaluation:
- **Processing Speed**: 24-28 FPS on consumer hardware
- **Form Detection Accuracy**: 91-96% for common errors
- **Exercise Coverage**: 3 foundational bodyweight exercises
- **Real-time Latency**: <50ms feedback delay

## 🎯 Exercise Analysis Details

### Squat Analysis
- **Depth Check**: Hip crease below knee level
- **Back Posture**: Maintains neutral spine alignment
- **Knee Tracking**: Proper knee-foot alignment
- **Common Errors**: Insufficient depth, forward lean, knee valgus

### Push-up Analysis
- **Body Alignment**: Straight line from head to heels
- **Range of Motion**: 90-degree elbow flexion minimum
- **Core Engagement**: Hip stability throughout movement
- **Common Errors**: Hip sag, incomplete range, elbow flaring

### Lunge Analysis
- **Knee Angles**: Both knees at ~90 degrees at bottom
- **Front Knee Position**: Knee behind toes
- **Balance**: Torso remains upright
- **Common Errors**: Insufficient depth, knee collapse, forward lean

## 🔧 Configuration Options

### MediaPipe Settings
- Detection confidence threshold (0.5-1.0)
- Tracking confidence threshold (0.5-1.0)
- Model complexity (Lite/Full/Heavy)

### Analysis Modes (ML Version)
- **Rule-based**: Traditional geometric analysis
- **ML-based**: Machine learning predictions
- **Hybrid**: Combined approach for enhanced accuracy

## 📈 Future Enhancements

### Planned Features
- Audio feedback integration (TTS)
- Additional exercise library expansion
- Mobile app deployment
- 3D pose estimation integration
- Personalized coaching algorithms

### Research Applications
- Biomechanical research data collection
- Exercise science studies
- Rehabilitation monitoring
- Sports performance analysis

## 🏆 Academic Context

This project implements the comprehensive research framework described in:
- **Institution**: Chandigarh Engineering College, Department of AI & ML
- **Authors**: Raj Pratap Singh, Kunal Kashyap, Kameshwar Singh
- **Supervision**: Dr. Ishu
- **Research Focus**: Real-time vision-based biomechanical feedback systems

## 📚 Technical References

### Core Technologies
- [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
- [OpenCV Python](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [NumPy](https://numpy.org/doc/stable/)
- [Tkinter](https://docs.python.org/3/library/tkinter.html)

### Biomechanical Principles
- Exercise technique standards from NSCA and ACE
- Kinesiology and movement science foundations
- Joint angle analysis and postural assessment

## 🤝 Contributing

This project welcomes contributions in the following areas:
- Additional exercise implementations
- ML model improvements
- UI/UX enhancements
- Performance optimizations
- Documentation improvements

## 📄 License

This project is developed for academic and educational purposes. Please refer to the specific license terms for commercial use.

## 📞 Support

For technical support or research collaboration inquiries, please refer to the academic institution contact information.

---

**Note**: This application is designed for fitness guidance and educational purposes. It is not intended as a substitute for professional medical advice or physical therapy. Users with pre-existing conditions should consult healthcare professionals before beginning any exercise program.
'''

with open("README.md", "w") as f:
    f.write(readme_content)

# Create a setup script
setup_script = '''#!/usr/bin/env python3
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
    
    print("\\n🎉 Setup completed successfully!")
    print("\\nTo run the application:")
    print("  Basic version: python ai_workout_form_corrector.py")
    print("  ML version:    python ai_workout_form_corrector_ml.py")

if __name__ == "__main__":
    main()
'''

with open("setup.py", "w") as f:
    f.write(setup_script)

# Make setup script executable (Unix-like systems)
try:
    os.chmod("setup.py", 0o755)
except:
    pass  # Windows doesn't need this

# Create a configuration file template
config_template = '''{
    "mediapipe_settings": {
        "min_detection_confidence": 0.7,
        "min_tracking_confidence": 0.7,
        "model_complexity": 1
    },
    "exercise_thresholds": {
        "squat": {
            "depth_angle": 100,
            "back_angle": 80,
            "stage_transition": 120
        },
        "pushup": {
            "elbow_angle": 95,
            "body_alignment": 160,
            "stage_transition": 140
        },
        "lunge": {
            "knee_angle": 110,
            "knee_position_tolerance": 50,
            "stage_transition": 140
        }
    },
    "ui_settings": {
        "video_width": 640,
        "video_height": 480,
        "update_interval": 15,
        "theme": "dark"
    },
    "ml_settings": {
        "model_type": "RandomForest",
        "n_estimators": 100,
        "feature_count": 15,
        "training_split": 0.8
    }
}'''

with open("config.json", "w") as f:
    f.write(config_template)

# Create a simple test script
test_script = '''#!/usr/bin/env python3
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
    print("\\n📹 Testing camera functionality...")
    
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
    print("\\n🤖 Testing MediaPipe pose estimation...")
    
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
        print(f"\\n--- {test_name} ---")
        if test_func():
            passed += 1
    
    print(f"\\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
        return 0
    else:
        print("❌ Some tests failed. Check the setup.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
'''

with open("test_system.py", "w") as f:
    f.write(test_script)

print("✅ Created documentation and setup files:")
print("  - README.md")
print("  - setup.py") 
print("  - config.json")
print("  - test_system.py")