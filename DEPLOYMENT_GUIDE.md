# AI Workout Form Corrector - Deployment Guide

## Quick Start (5 Minutes)

### Step 1: System Requirements Check
```bash
# Verify Python version (3.9+ required)
python --version

# Check camera access
# Run test script after installation
python test_system.py
```

### Step 2: Installation
```bash
# Basic installation
pip install -r requirements.txt

# Full ML installation (recommended)
pip install -r requirements_full.txt

# Run setup script
python setup.py
```

### Step 3: Launch Application
```bash
# Basic version (lightweight)
python ai_workout_form_corrector.py

# ML-enhanced version (full features)
python ai_workout_form_corrector_ml.py
```

## Detailed Setup Instructions

### Environment Setup

#### Windows
```cmd
# Install Python 3.9+ from python.org
# Open Command Prompt
cd path\to\project
pip install -r requirements_full.txt
python ai_workout_form_corrector_ml.py
```

#### macOS
```bash
# Install via Homebrew
brew install python@3.9
pip3 install -r requirements_full.txt
python3 ai_workout_form_corrector_ml.py
```

#### Linux (Ubuntu/Debian)
```bash
# Install dependencies
sudo apt update
sudo apt install python3.9 python3-pip python3-tk
pip3 install -r requirements_full.txt
python3 ai_workout_form_corrector_ml.py
```

### Camera Setup

1. **Position**: 3-6 feet from camera
2. **Lighting**: Ensure adequate, even lighting
3. **Background**: Minimal clutter, contrasting background
4. **View**: For squats/lunges use side view; push-ups can use angle view

### Performance Optimization

#### For Better FPS
- Close unnecessary applications
- Use good lighting conditions
- Position camera at eye level
- Ensure stable internet (if using cloud features)

#### For Better Accuracy
- Wear contrasting clothing
- Ensure full body visible
- Maintain consistent lighting
- Use recommended camera angles

## Troubleshooting

### Common Issues

#### "Camera not accessible"
```bash
# Check camera permissions
# Windows: Settings > Privacy > Camera
# macOS: System Preferences > Security & Privacy > Camera
# Linux: Check /dev/video* permissions
```

#### "MediaPipe import failed"
```bash
# Reinstall MediaPipe
pip uninstall mediapipe
pip install mediapipe==0.8.11
```

#### Low FPS Performance
```bash
# Reduce video resolution in config.json
# Close other camera applications
# Update graphics drivers
```

#### ML Model Errors
```bash
# Reinstall scikit-learn
pip uninstall scikit-learn
pip install scikit-learn==1.0.2
```

### Hardware Requirements

#### Minimum Specifications
- CPU: Intel i3 / AMD Ryzen 3 equivalent
- RAM: 4GB
- Camera: 720p webcam
- OS: Windows 10, macOS 10.14, Ubuntu 18.04

#### Recommended Specifications
- CPU: Intel i5 / AMD Ryzen 5 equivalent
- RAM: 8GB
- Camera: 1080p webcam with good low-light performance
- OS: Latest versions

## Configuration Options

### MediaPipe Settings (config.json)
```json
{
  "mediapipe_settings": {
    "min_detection_confidence": 0.7,  # Higher = more strict
    "min_tracking_confidence": 0.7,   # Higher = more stable
    "model_complexity": 1             # 0=Lite, 1=Full, 2=Heavy
  }
}
```

### Exercise Thresholds
```json
{
  "exercise_thresholds": {
    "squat": {
      "depth_angle": 100,      # Minimum hip angle for depth
      "back_angle": 80,        # Minimum back straightness
      "stage_transition": 120  # Angle for up/down detection
    }
  }
}
```

## Advanced Features

### Data Export
1. Use "Export Data" button in Analytics tab
2. Saves CSV with workout history
3. Import into Excel/Google Sheets for analysis

### Model Training
1. Use "Record Data" during workouts
2. Label good/bad form examples
3. Retrain models in ML Training tab
4. Save improved models for future use

### Custom Exercise Addition
1. Study existing exercise analysis functions
2. Implement new biomechanical rules
3. Add to exercise selection menu
4. Test thoroughly before deployment

## Production Deployment

### For Research/Educational Use
- No additional licensing required
- Cite original academic work
- Document any modifications

### For Commercial Use
- Review MediaPipe licensing
- Consider trademark/patent implications
- Implement proper error handling
- Add comprehensive logging

### For Clinical/Medical Use
- Consult healthcare regulations
- Implement data privacy measures
- Add medical disclaimers
- Consider FDA/regulatory approval

## Support and Maintenance

### Regular Updates
```bash
# Update dependencies monthly
pip install -r requirements_full.txt --upgrade

# Check for MediaPipe updates
pip install mediapipe --upgrade
```

### Performance Monitoring
- Monitor FPS in application
- Track form detection accuracy
- Review user feedback regularly
- Update thresholds based on usage data

### Data Backup
- Export analytics data regularly
- Backup trained ML models
- Save configuration files
- Document customizations

## Integration Options

### API Development
- Create REST API wrapper
- Enable remote exercise analysis
- Implement cloud data storage
- Add multi-user support

### Mobile Integration
- Port to React Native/Flutter
- Use TensorFlow Lite for mobile
- Optimize for mobile cameras
- Add offline capabilities

### Web Application
- Convert to web-based system
- Use TensorFlow.js for browser
- Implement WebRTC for video
- Add cloud analytics dashboard
