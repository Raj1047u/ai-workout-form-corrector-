# Create additional ML utility files and enhanced requirements

# Enhanced requirements with ML packages
enhanced_requirements = '''# Core Computer Vision and GUI
opencv-python>=4.5.0
mediapipe>=0.8.0
numpy>=1.21.0
Pillow>=8.3.0

# Machine Learning
scikit-learn>=1.0.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0

# GUI and Visualization
tkinter
ttkthemes>=3.2.0

# Data Processing
scipy>=1.7.0
joblib>=1.1.0

# Optional: Advanced ML (uncomment if needed)
# tensorflow>=2.8.0
# torch>=1.11.0
# xgboost>=1.5.0

# Development and Testing
pytest>=6.2.0
pytest-cov>=2.12.0

# Documentation
sphinx>=4.0.0
'''

with open("requirements_full.txt", "w") as f:
    f.write(enhanced_requirements.strip())

# Create ML utilities module
ml_utils_code = '''"""
ML Utilities for AI Workout Form Corrector
Advanced machine learning features and data processing utilities
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import json
from datetime import datetime
import os

class PoseFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Advanced feature extractor for pose landmarks
    Converts raw landmark coordinates into meaningful biomechanical features
    """
    
    def __init__(self, include_temporal=True, window_size=5):
        self.include_temporal = include_temporal
        self.window_size = window_size
        self.landmark_history = []
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Transform landmark data into features"""
        if isinstance(X, np.ndarray):
            return self._extract_features_array(X)
        elif isinstance(X, list):
            return self._extract_features_list(X)
        else:
            raise ValueError("Input must be numpy array or list")
    
    def _extract_features_array(self, landmarks_array):
        """Extract features from numpy array of landmarks"""
        features_list = []
        
        for landmarks in landmarks_array:
            features = self._extract_single_frame_features(landmarks)
            features_list.append(features)
        
        return np.array(features_list)
    
    def _extract_features_list(self, landmarks_list):
        """Extract features from list of landmark frames"""
        return self._extract_features_array(landmarks_list)
    
    def _extract_single_frame_features(self, landmarks):
        """Extract comprehensive features from single frame"""
        if landmarks is None or len(landmarks) < 33:
            return np.zeros(25)  # Return zero features if invalid input
        
        features = []
        
        try:
            # Extract key points
            key_points = self._get_key_points(landmarks)
            
            # 1. Joint angles (8 features)
            angles = self._calculate_joint_angles(key_points)
            features.extend(angles)
            
            # 2. Distance ratios (5 features)
            distances = self._calculate_distance_ratios(key_points)
            features.extend(distances)
            
            # 3. Symmetry measures (4 features)
            symmetry = self._calculate_symmetry(landmarks)
            features.extend(symmetry)
            
            # 4. Stability measures (3 features)
            stability = self._calculate_stability(landmarks)
            features.extend(stability)
            
            # 5. Body alignment (3 features)
            alignment = self._calculate_alignment(key_points)
            features.extend(alignment)
            
            # 6. Velocity features (2 features) - if temporal data available
            if self.include_temporal and len(self.landmark_history) > 0:
                velocity = self._calculate_velocity(landmarks)
                features.extend(velocity)
            else:
                features.extend([0, 0])  # Zero velocity if no history
            
            # Update history
            if self.include_temporal:
                self.landmark_history.append(landmarks)
                if len(self.landmark_history) > self.window_size:
                    self.landmark_history.pop(0)
                    
        except Exception as e:
            print(f"Feature extraction error: {e}")
            features = [0] * 25  # Return zero features on error
        
        return np.array(features[:25])  # Ensure exactly 25 features
    
    def _get_key_points(self, landmarks):
        """Extract key anatomical points"""
        return {
            'nose': landmarks[0][:2],
            'left_shoulder': landmarks[11][:2],
            'right_shoulder': landmarks[12][:2],
            'left_elbow': landmarks[13][:2],
            'right_elbow': landmarks[14][:2],
            'left_wrist': landmarks[15][:2],
            'right_wrist': landmarks[16][:2],
            'left_hip': landmarks[23][:2],
            'right_hip': landmarks[24][:2],
            'left_knee': landmarks[25][:2],
            'right_knee': landmarks[26][:2],
            'left_ankle': landmarks[27][:2],
            'right_ankle': landmarks[28][:2],
        }
    
    def _calculate_joint_angles(self, key_points):
        """Calculate important joint angles"""
        angles = []
        
        try:
            # Shoulder center and hip center
            shoulder_center = np.mean([key_points['left_shoulder'], key_points['right_shoulder']], axis=0)
            hip_center = np.mean([key_points['left_hip'], key_points['right_hip']], axis=0)
            knee_center = np.mean([key_points['left_knee'], key_points['right_knee']], axis=0)
            ankle_center = np.mean([key_points['left_ankle'], key_points['right_ankle']], axis=0)
            
            # Hip angle (shoulder-hip-knee)
            hip_angle = self._angle_between_points(shoulder_center, hip_center, knee_center)
            angles.append(hip_angle)
            
            # Knee angle (hip-knee-ankle)
            knee_angle = self._angle_between_points(hip_center, knee_center, ankle_center)
            angles.append(knee_angle)
            
            # Left elbow angle
            left_elbow_angle = self._angle_between_points(
                key_points['left_shoulder'], key_points['left_elbow'], key_points['left_wrist']
            )
            angles.append(left_elbow_angle)
            
            # Right elbow angle
            right_elbow_angle = self._angle_between_points(
                key_points['right_shoulder'], key_points['right_elbow'], key_points['right_wrist']
            )
            angles.append(right_elbow_angle)
            
            # Spine angle (approximated)
            spine_angle = self._angle_between_points(shoulder_center, hip_center, ankle_center)
            angles.append(spine_angle)
            
            # Left knee angle
            left_knee_angle = self._angle_between_points(
                key_points['left_hip'], key_points['left_knee'], key_points['left_ankle']
            )
            angles.append(left_knee_angle)
            
            # Right knee angle
            right_knee_angle = self._angle_between_points(
                key_points['right_hip'], key_points['right_knee'], key_points['right_ankle']
            )
            angles.append(right_knee_angle)
            
            # Torso angle (vertical reference)
            torso_angle = self._vertical_angle(shoulder_center, hip_center)
            angles.append(torso_angle)
            
        except Exception as e:
            angles = [0] * 8
        
        return angles[:8]  # Ensure exactly 8 angles
    
    def _calculate_distance_ratios(self, key_points):
        """Calculate important distance ratios for normalization"""
        ratios = []
        
        try:
            # Shoulder width (reference distance)
            shoulder_width = np.linalg.norm(
                np.array(key_points['left_shoulder']) - np.array(key_points['right_shoulder'])
            )
            
            if shoulder_width == 0:
                return [0] * 5
            
            # Hip width to shoulder width ratio
            hip_width = np.linalg.norm(
                np.array(key_points['left_hip']) - np.array(key_points['right_hip'])
            )
            ratios.append(hip_width / shoulder_width)
            
            # Torso length to shoulder width ratio
            shoulder_center = np.mean([key_points['left_shoulder'], key_points['right_shoulder']], axis=0)
            hip_center = np.mean([key_points['left_hip'], key_points['right_hip']], axis=0)
            torso_length = np.linalg.norm(shoulder_center - hip_center)
            ratios.append(torso_length / shoulder_width)
            
            # Leg length to shoulder width ratio
            knee_center = np.mean([key_points['left_knee'], key_points['right_knee']], axis=0)
            ankle_center = np.mean([key_points['left_ankle'], key_points['right_ankle']], axis=0)
            upper_leg = np.linalg.norm(hip_center - knee_center)
            lower_leg = np.linalg.norm(knee_center - ankle_center)
            leg_length = upper_leg + lower_leg
            ratios.append(leg_length / shoulder_width)
            
            # Arm length to shoulder width ratio
            left_arm = (np.linalg.norm(np.array(key_points['left_shoulder']) - np.array(key_points['left_elbow'])) +
                       np.linalg.norm(np.array(key_points['left_elbow']) - np.array(key_points['left_wrist'])))
            ratios.append(left_arm / shoulder_width)
            
            # Height to width ratio
            total_height = np.linalg.norm(shoulder_center - ankle_center)
            ratios.append(total_height / shoulder_width)
            
        except Exception as e:
            ratios = [1.0] * 5  # Default ratios
        
        return ratios[:5]
    
    def _calculate_symmetry(self, landmarks):
        """Calculate left-right symmetry measures"""
        symmetry = []
        
        try:
            # Hip height symmetry
            left_hip_y = landmarks[23][1]
            right_hip_y = landmarks[24][1]
            hip_symmetry = abs(left_hip_y - right_hip_y)
            symmetry.append(hip_symmetry)
            
            # Knee height symmetry
            left_knee_y = landmarks[25][1]
            right_knee_y = landmarks[26][1]
            knee_symmetry = abs(left_knee_y - right_knee_y)
            symmetry.append(knee_symmetry)
            
            # Shoulder height symmetry
            left_shoulder_y = landmarks[11][1]
            right_shoulder_y = landmarks[12][1]
            shoulder_symmetry = abs(left_shoulder_y - right_shoulder_y)
            symmetry.append(shoulder_symmetry)
            
            # Overall body symmetry (center line deviation)
            center_x = (landmarks[11][0] + landmarks[12][0]) / 2  # Shoulder center x
            hip_center_x = (landmarks[23][0] + landmarks[24][0]) / 2  # Hip center x
            center_deviation = abs(center_x - hip_center_x)
            symmetry.append(center_deviation)
            
        except Exception as e:
            symmetry = [0] * 4
        
        return symmetry[:4]
    
    def _calculate_stability(self, landmarks):
        """Calculate pose stability measures"""
        stability = []
        
        try:
            # Visibility score average
            visibility_scores = [lm[2] if len(lm) > 2 else 0.5 for lm in landmarks]
            avg_visibility = np.mean(visibility_scores)
            stability.append(avg_visibility)
            
            # Joint confidence variance (measure of stability)
            visibility_variance = np.var(visibility_scores)
            stability.append(1.0 - min(visibility_variance, 1.0))  # Invert so high variance = low stability
            
            # Key joint visibility (important joints)
            key_joints = [11, 12, 23, 24, 25, 26]  # Shoulders, hips, knees
            key_visibility = np.mean([landmarks[i][2] if len(landmarks[i]) > 2 else 0.5 for i in key_joints])
            stability.append(key_visibility)
            
        except Exception as e:
            stability = [0.5] * 3
        
        return stability[:3]
    
    def _calculate_alignment(self, key_points):
        """Calculate body alignment measures"""
        alignment = []
        
        try:
            # Vertical alignment (shoulder-hip-ankle)
            shoulder_center = np.mean([key_points['left_shoulder'], key_points['right_shoulder']], axis=0)
            hip_center = np.mean([key_points['left_hip'], key_points['right_hip']], axis=0)
            ankle_center = np.mean([key_points['left_ankle'], key_points['right_ankle']], axis=0)
            
            # Calculate how much hip deviates from shoulder-ankle line
            alignment_deviation = self._point_line_distance(hip_center, shoulder_center, ankle_center)
            alignment.append(alignment_deviation)
            
            # Forward/backward lean
            shoulder_ankle_vector = ankle_center - shoulder_center
            lean_angle = np.arctan2(shoulder_ankle_vector[0], shoulder_ankle_vector[1]) * 180 / np.pi
            alignment.append(abs(lean_angle))
            
            # Left/right lean
            side_lean = abs(shoulder_center[0] - ankle_center[0])
            alignment.append(side_lean)
            
        except Exception as e:
            alignment = [0] * 3
        
        return alignment[:3]
    
    def _calculate_velocity(self, current_landmarks):
        """Calculate velocity features from temporal data"""
        if len(self.landmark_history) == 0:
            return [0, 0]
        
        try:
            prev_landmarks = self.landmark_history[-1]
            
            # Hip center velocity
            curr_hip = np.mean([current_landmarks[23][:2], current_landmarks[24][:2]], axis=0)
            prev_hip = np.mean([prev_landmarks[23][:2], prev_landmarks[24][:2]], axis=0)
            hip_velocity = np.linalg.norm(curr_hip - prev_hip)
            
            # Overall body movement (average of key points)
            key_indices = [11, 12, 23, 24, 25, 26]  # Shoulders, hips, knees
            velocities = []
            
            for idx in key_indices:
                curr_point = np.array(current_landmarks[idx][:2])
                prev_point = np.array(prev_landmarks[idx][:2])
                velocity = np.linalg.norm(curr_point - prev_point)
                velocities.append(velocity)
            
            avg_velocity = np.mean(velocities)
            
            return [hip_velocity, avg_velocity]
            
        except Exception as e:
            return [0, 0]
    
    def _angle_between_points(self, a, b, c):
        """Calculate angle at point b formed by points a, b, c"""
        try:
            a, b, c = np.array(a), np.array(b), np.array(c)
            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(radians * 180.0 / np.pi)
            return 360 - angle if angle > 180.0 else angle
        except:
            return 0
    
    def _vertical_angle(self, point1, point2):
        """Calculate angle from vertical"""
        try:
            vector = np.array(point2) - np.array(point1)
            angle = np.arctan2(abs(vector[0]), abs(vector[1])) * 180 / np.pi
            return angle
        except:
            return 0
    
    def _point_line_distance(self, point, line_point1, line_point2):
        """Calculate distance from point to line"""
        try:
            p, p1, p2 = np.array(point), np.array(line_point1), np.array(line_point2)
            
            # Line vector
            line_vec = p2 - p1
            line_len = np.linalg.norm(line_vec)
            
            if line_len == 0:
                return np.linalg.norm(p - p1)
            
            # Point vector
            point_vec = p - p1
            
            # Project point onto line
            projection = np.dot(point_vec, line_vec) / line_len
            projection_point = p1 + (projection / line_len) * line_vec
            
            # Distance from point to projection
            distance = np.linalg.norm(p - projection_point)
            return distance
            
        except:
            return 0

class ExerciseClassifier:
    """
    Advanced exercise classifier with ensemble methods
    """
    
    def __init__(self, exercise_type, model_config=None):
        self.exercise_type = exercise_type
        self.model_config = model_config or {}
        self.feature_extractor = PoseFeatureExtractor()
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False
        
    def train(self, X, y, validation_split=0.2):
        """Train the classifier"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        # Extract features
        X_features = self.feature_extractor.fit_transform(X)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_features, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=self.model_config.get('n_estimators', 100),
            max_depth=self.model_config.get('max_depth', None),
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        val_predictions = self.model.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val, val_predictions)
        
        self.is_trained = True
        
        return {
            'validation_accuracy': val_accuracy,
            'training_samples': len(X_train),
            'validation_samples': len(X_val)
        }
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_features = self.feature_extractor.transform(X)
        X_scaled = self.scaler.transform(X_features.reshape(1, -1))
        
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]
        confidence = max(probability)
        
        return prediction, confidence
    
    def save(self, filepath):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_extractor': self.feature_extractor,
            'exercise_type': self.exercise_type,
            'model_config': self.model_config
        }
        
        joblib.dump(model_data, filepath)
    
    def load(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_extractor = model_data['feature_extractor']
        self.exercise_type = model_data['exercise_type']
        self.model_config = model_data['model_config']
        self.is_trained = True

class WorkoutDataManager:
    """
    Manage workout data, sessions, and analytics
    """
    
    def __init__(self, data_dir="workout_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.sessions_file = os.path.join(data_dir, "sessions.json")
        self.analytics_file = os.path.join(data_dir, "analytics.json")
        
    def save_session(self, session_data):
        """Save a workout session"""
        sessions = self.load_sessions()
        
        session_entry = {
            'timestamp': datetime.now().isoformat(),
            'exercise': session_data.get('exercise', 'unknown'),
            'repetitions': session_data.get('repetitions', 0),
            'duration_seconds': session_data.get('duration', 0),
            'average_form_score': session_data.get('form_score', 0.0),
            'feedback_count': session_data.get('feedback_count', 0),
            'pose_data': session_data.get('pose_data', [])
        }
        
        sessions.append(session_entry)
        
        with open(self.sessions_file, 'w') as f:
            json.dump(sessions, f, indent=2)
    
    def load_sessions(self):
        """Load all workout sessions"""
        if os.path.exists(self.sessions_file):
            with open(self.sessions_file, 'r') as f:
                return json.load(f)
        return []
    
    def get_analytics(self, days=30):
        """Get workout analytics for specified days"""
        sessions = self.load_sessions()
        
        if not sessions:
            return {}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(sessions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter recent sessions
        recent_date = datetime.now() - pd.Timedelta(days=days)
        recent_sessions = df[df['timestamp'] >= recent_date]
        
        if recent_sessions.empty:
            return {}
        
        analytics = {
            'total_sessions': len(recent_sessions),
            'total_repetitions': recent_sessions['repetitions'].sum(),
            'average_session_duration': recent_sessions['duration_seconds'].mean(),
            'average_form_score': recent_sessions['average_form_score'].mean(),
            'exercises_breakdown': recent_sessions['exercise'].value_counts().to_dict(),
            'daily_activity': recent_sessions.groupby(recent_sessions['timestamp'].dt.date).size().to_dict(),
            'progress_trend': recent_sessions.groupby(recent_sessions['timestamp'].dt.date)['average_form_score'].mean().to_dict()
        }
        
        return analytics
    
    def export_data(self, filepath, format='csv'):
        """Export workout data"""
        sessions = self.load_sessions()
        
        if not sessions:
            raise ValueError("No session data to export")
        
        df = pd.DataFrame(sessions)
        
        if format.lower() == 'csv':
            df.to_csv(filepath, index=False)
        elif format.lower() == 'json':
            df.to_json(filepath, orient='records', indent=2)
        else:
            raise ValueError("Supported formats: csv, json")

# Example usage and testing
if __name__ == "__main__":
    print("ML Utilities for AI Workout Form Corrector")
    print("Testing feature extraction...")
    
    # Create sample landmarks data
    sample_landmarks = []
    for i in range(33):
        sample_landmarks.append([100 + i*10, 200 + i*5, 0.8])  # x, y, visibility
    
    # Test feature extractor
    extractor = PoseFeatureExtractor()
    features = extractor.transform([sample_landmarks])
    
    print(f"Extracted {features.shape[1]} features from pose data")
    print(f"Features shape: {features.shape}")
    print("✅ Feature extraction test passed")
'''

with open("ml_utils.py", "w") as f:
    f.write(ml_utils_code)

print("✅ Created additional ML files:")
print("  - requirements_full.txt (enhanced dependencies)")
print("  - ml_utils.py (ML utilities and feature extraction)")