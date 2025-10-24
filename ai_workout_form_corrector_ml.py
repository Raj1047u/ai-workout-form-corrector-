
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import threading
import time
import json
import pickle
from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkinter

class MLWorkoutFormCorrector:
    """Enhanced AI Workout Form Corrector with Machine Learning capabilities"""

    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            static_image_mode=False
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # ML Models for each exercise
        self.ml_models = {
            'squat': None,
            'pushup': None,
            'lunge': None
        }
        self.scalers = {
            'squat': StandardScaler(),
            'pushup': StandardScaler(),
            'lunge': StandardScaler()
        }

        # Data collection for training
        self.training_data = {
            'squat': {'features': [], 'labels': []},
            'pushup': {'features': [], 'labels': []},
            'lunge': {'features': [], 'labels': []}
        }

        # Initialize base class attributes
        self.cap = None
        self.current_exercise = "squat"
        self.stage = "up"
        self.counter = 0
        self.feedback = []
        self.form_ok = True
        self.is_running = False
        self.is_recording = False

        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

        # Session data for analytics
        self.session_data = []
        self.workout_history = []

        # GUI variables
        self.root = None
        self.notebook = None

        # Initialize ML models
        self.initialize_ml_models()

        # Setup GUI
        self.setup_enhanced_gui()

    def initialize_ml_models(self):
        """Initialize machine learning models with sample data"""
        # Create sample training data for demonstration
        np.random.seed(42)

        for exercise in ['squat', 'pushup', 'lunge']:
            # Generate synthetic features for demonstration
            n_samples = 1000
            n_features = 15  # Angle features, velocity, acceleration

            # Good form samples (label = 1)
            good_features = np.random.normal(0, 1, (n_samples//2, n_features))
            good_labels = np.ones(n_samples//2)

            # Poor form samples (label = 0)
            poor_features = np.random.normal(0, 2, (n_samples//2, n_features))  # More variance
            poor_labels = np.zeros(n_samples//2)

            # Combine data
            X = np.vstack([good_features, poor_features])
            y = np.hstack([good_labels, poor_labels])

            # Train model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Scale features
            self.scalers[exercise].fit(X_train)
            X_train_scaled = self.scalers[exercise].transform(X_train)
            X_test_scaled = self.scalers[exercise].transform(X_test)

            # Train Random Forest model
            self.ml_models[exercise] = RandomForestClassifier(n_estimators=100, random_state=42)
            self.ml_models[exercise].fit(X_train_scaled, y_train)

            # Evaluate model
            y_pred = self.ml_models[exercise].predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"✅ {exercise.title()} ML model trained - Accuracy: {accuracy:.2f}")

    def extract_advanced_features(self, landmarks, prev_landmarks=None):
        """Extract advanced features for ML analysis"""
        if not landmarks:
            return np.zeros(15)  # Return zero vector if no landmarks

        features = []

        try:
            # Key landmarks
            shoulder = np.mean([landmarks[11][:2], landmarks[12][:2]], axis=0)
            hip = np.mean([landmarks[23][:2], landmarks[24][:2]], axis=0)
            knee = np.mean([landmarks[25][:2], landmarks[26][:2]], axis=0)
            ankle = np.mean([landmarks[27][:2], landmarks[28][:2]], axis=0)
            elbow = np.mean([landmarks[13][:2], landmarks[14][:2]], axis=0)
            wrist = np.mean([landmarks[15][:2], landmarks[16][:2]], axis=0)

            # Angle features
            hip_angle = self.calculate_angle(shoulder, hip, knee)
            knee_angle = self.calculate_angle(hip, knee, ankle)
            elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
            back_angle = self.calculate_angle(shoulder, hip, ankle)
            body_angle = self.calculate_angle(shoulder, hip, ankle)

            # Add angles to features
            features.extend([hip_angle, knee_angle, elbow_angle, back_angle, body_angle])

            # Distance features
            shoulder_hip_dist = np.linalg.norm(shoulder - hip)
            hip_knee_dist = np.linalg.norm(hip - knee)
            knee_ankle_dist = np.linalg.norm(knee - ankle)

            features.extend([shoulder_hip_dist, hip_knee_dist, knee_ankle_dist])

            # Symmetry features
            left_hip = landmarks[23][:2]
            right_hip = landmarks[24][:2]
            left_knee = landmarks[25][:2]
            right_knee = landmarks[26][:2]

            hip_symmetry = abs(left_hip[1] - right_hip[1])  # Y-coordinate difference
            knee_symmetry = abs(left_knee[1] - right_knee[1])

            features.extend([hip_symmetry, knee_symmetry])

            # Velocity features (if previous landmarks available)
            if prev_landmarks:
                prev_hip = np.mean([prev_landmarks[23][:2], prev_landmarks[24][:2]], axis=0)
                hip_velocity = np.linalg.norm(hip - prev_hip)
            else:
                hip_velocity = 0

            features.extend([hip_velocity])

            # Body alignment features
            alignment_score = self.calculate_alignment_score(landmarks)
            stability_score = self.calculate_stability_score(landmarks)

            features.extend([alignment_score, stability_score])

        except Exception as e:
            # Return zero features if extraction fails
            features = [0] * 15

        return np.array(features[:15])  # Ensure exactly 15 features

    def calculate_alignment_score(self, landmarks):
        """Calculate body alignment score"""
        try:
            shoulder = np.mean([landmarks[11][:2], landmarks[12][:2]], axis=0)
            hip = np.mean([landmarks[23][:2], landmarks[24][:2]], axis=0)
            ankle = np.mean([landmarks[27][:2], landmarks[28][:2]], axis=0)

            # Calculate deviation from straight line
            expected_y = shoulder[1] + (ankle[1] - shoulder[1]) * (hip[0] - shoulder[0]) / (ankle[0] - shoulder[0])
            deviation = abs(hip[1] - expected_y)

            # Convert to score (lower deviation = higher score)
            alignment_score = max(0, 100 - deviation / 10)
            return alignment_score
        except:
            return 50  # Neutral score if calculation fails

    def calculate_stability_score(self, landmarks):
        """Calculate pose stability score"""
        try:
            # Use landmark visibility scores as stability indicators
            visibility_scores = [lm[2] for lm in landmarks if len(lm) > 2]
            if visibility_scores:
                return np.mean(visibility_scores) * 100
            return 50
        except:
            return 50

    def ml_analyze_form(self, landmarks, prev_landmarks=None):
        """Use ML model to analyze exercise form"""
        if self.ml_models[self.current_exercise] is None:
            return "Rule-based analysis", 0.5

        # Extract features
        features = self.extract_advanced_features(landmarks, prev_landmarks)
        features = features.reshape(1, -1)

        # Scale features
        features_scaled = self.scalers[self.current_exercise].transform(features)

        # Predict form quality
        prediction = self.ml_models[self.current_exercise].predict(features_scaled)[0]
        probability = self.ml_models[self.current_exercise].predict_proba(features_scaled)[0]

        # Get confidence score
        confidence = max(probability)

        if prediction == 1:
            return "Excellent form!", confidence
        else:
            return "Form needs improvement", confidence

    def setup_enhanced_gui(self):
        """Setup enhanced GUI with tabbed interface"""
        self.root = tk.Tk()
        self.root.title("AI Workout Form Corrector - ML Enhanced")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')

        # Create notebook for tabs
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook.Tab', padding=[12, 8])

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tabs
        self.create_workout_tab()
        self.create_analytics_tab()
        self.create_training_tab()
        self.create_settings_tab()

    def create_workout_tab(self):
        """Create main workout tab"""
        workout_frame = tk.Frame(self.notebook, bg='#2c3e50')
        self.notebook.add(workout_frame, text='Workout')

        # Main container
        main_frame = tk.Frame(workout_frame, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Video frame (left side)
        video_frame = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Video label
        self.video_label = tk.Label(video_frame, bg='black', text="Video Feed\nPress Start to begin", 
                                  font=('Arial', 16), fg='white')
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Control panel (right side)
        control_frame = tk.Frame(main_frame, bg='#34495e', width=350, relief=tk.RAISED, bd=2)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        control_frame.pack_propagate(False)

        # Title
        title_label = tk.Label(control_frame, text="AI Workout Form Corrector", 
                             font=('Arial', 16, 'bold'), bg='#34495e', fg='white')
        title_label.pack(pady=(10, 20))

        # Exercise selection
        exercise_frame = tk.LabelFrame(control_frame, text="Exercise Selection", 
                                     font=('Arial', 12, 'bold'), bg='#34495e', fg='white')
        exercise_frame.pack(fill=tk.X, padx=10, pady=5)

        self.exercise_var = tk.StringVar(value="squat")
        exercises = [("Squats", "squat"), ("Push-ups", "pushup"), ("Lunges", "lunge")]

        for text, value in exercises:
            rb = tk.Radiobutton(exercise_frame, text=text, variable=self.exercise_var, 
                              value=value, bg='#34495e', fg='white', font=('Arial', 10),
                              selectcolor='#2c3e50', command=self.change_exercise)
            rb.pack(anchor=tk.W, padx=10, pady=2)

        # Analysis mode selection
        mode_frame = tk.LabelFrame(control_frame, text="Analysis Mode", 
                                 font=('Arial', 12, 'bold'), bg='#34495e', fg='white')
        mode_frame.pack(fill=tk.X, padx=10, pady=5)

        self.analysis_mode = tk.StringVar(value="hybrid")
        modes = [("Rule-based", "rule"), ("ML-based", "ml"), ("Hybrid", "hybrid")]

        for text, value in modes:
            rb = tk.Radiobutton(mode_frame, text=text, variable=self.analysis_mode, 
                              value=value, bg='#34495e', fg='white', font=('Arial', 9),
                              selectcolor='#2c3e50')
            rb.pack(anchor=tk.W, padx=10, pady=1)

        # Control buttons
        button_frame = tk.Frame(control_frame, bg='#34495e')
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        self.start_button = tk.Button(button_frame, text="START WORKOUT", 
                                    command=self.start_workout,
                                    bg='#27ae60', fg='white', font=('Arial', 11, 'bold'),
                                    height=2)
        self.start_button.pack(fill=tk.X, pady=2)

        self.stop_button = tk.Button(button_frame, text="STOP WORKOUT", 
                                   command=self.stop_workout,
                                   bg='#e74c3c', fg='white', font=('Arial', 11, 'bold'),
                                   height=2, state=tk.DISABLED)
        self.stop_button.pack(fill=tk.X, pady=2)

        self.record_button = tk.Button(button_frame, text="RECORD DATA", 
                                     command=self.toggle_recording,
                                     bg='#f39c12', fg='white', font=('Arial', 11, 'bold'),
                                     height=2)
        self.record_button.pack(fill=tk.X, pady=2)

        # Statistics panel
        stats_frame = tk.LabelFrame(control_frame, text="Workout Statistics", 
                                  font=('Arial', 12, 'bold'), bg='#34495e', fg='white')
        stats_frame.pack(fill=tk.X, padx=10, pady=10)

        # Counter display
        tk.Label(stats_frame, text="Repetitions:", bg='#34495e', fg='white', 
                font=('Arial', 10)).pack(anchor=tk.W, padx=10)

        self.counter_label = tk.Label(stats_frame, text="0", bg='#34495e', fg='#f39c12', 
                                    font=('Arial', 24, 'bold'))
        self.counter_label.pack(anchor=tk.W, padx=10)

        # Performance metrics
        self.fps_label = tk.Label(stats_frame, text="FPS: 0", bg='#34495e', fg='white', 
                                font=('Arial', 10))
        self.fps_label.pack(anchor=tk.W, padx=10)

        self.ml_confidence_label = tk.Label(stats_frame, text="ML Confidence: N/A", 
                                          bg='#34495e', fg='white', font=('Arial', 10))
        self.ml_confidence_label.pack(anchor=tk.W, padx=10)

        # Feedback panel
        feedback_frame = tk.LabelFrame(control_frame, text="Real-time Feedback", 
                                     font=('Arial', 12, 'bold'), bg='#34495e', fg='white')
        feedback_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.feedback_text = tk.Text(feedback_frame, height=8, bg='#2c3e50', fg='white',
                                   font=('Arial', 10), wrap=tk.WORD, state=tk.DISABLED)
        self.feedback_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_analytics_tab(self):
        """Create analytics and progress tracking tab"""
        analytics_frame = tk.Frame(self.notebook, bg='#2c3e50')
        self.notebook.add(analytics_frame, text='Analytics')

        # Create matplotlib figure for progress charts
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.patch.set_facecolor('#2c3e50')

        self.ax1.set_facecolor('#34495e')
        self.ax2.set_facecolor('#34495e')

        self.canvas = FigureCanvasTkinter(self.fig, analytics_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control buttons for analytics
        analytics_controls = tk.Frame(analytics_frame, bg='#2c3e50')
        analytics_controls.pack(fill=tk.X, padx=10, pady=5)

        tk.Button(analytics_controls, text="Update Charts", command=self.update_analytics,
                 bg='#3498db', fg='white', font=('Arial', 10)).pack(side=tk.LEFT, padx=5)

        tk.Button(analytics_controls, text="Export Data", command=self.export_data,
                 bg='#9b59b6', fg='white', font=('Arial', 10)).pack(side=tk.LEFT, padx=5)

        self.update_analytics()

    def create_training_tab(self):
        """Create ML model training tab"""
        training_frame = tk.Frame(self.notebook, bg='#2c3e50')
        self.notebook.add(training_frame, text='ML Training')

        # Training controls
        control_frame = tk.LabelFrame(training_frame, text="Model Training Controls",
                                    bg='#34495e', fg='white', font=('Arial', 12, 'bold'))
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Button(control_frame, text="Retrain Models", command=self.retrain_models,
                 bg='#e67e22', fg='white', font=('Arial', 10)).pack(side=tk.LEFT, padx=5, pady=5)

        tk.Button(control_frame, text="Load Training Data", command=self.load_training_data,
                 bg='#16a085', fg='white', font=('Arial', 10)).pack(side=tk.LEFT, padx=5, pady=5)

        tk.Button(control_frame, text="Save Models", command=self.save_models,
                 bg='#8e44ad', fg='white', font=('Arial', 10)).pack(side=tk.LEFT, padx=5, pady=5)

        # Model performance display
        self.training_text = tk.Text(training_frame, bg='#2c3e50', fg='white',
                                   font=('Courier', 10))
        self.training_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def create_settings_tab(self):
        """Create settings and configuration tab"""
        settings_frame = tk.Frame(self.notebook, bg='#2c3e50')
        self.notebook.add(settings_frame, text='Settings')

        # MediaPipe settings
        mp_frame = tk.LabelFrame(settings_frame, text="MediaPipe Configuration",
                               bg='#34495e', fg='white', font=('Arial', 12, 'bold'))
        mp_frame.pack(fill=tk.X, padx=10, pady=10)

        # Detection confidence
        tk.Label(mp_frame, text="Detection Confidence:", bg='#34495e', fg='white').grid(row=0, column=0, padx=10, pady=5, sticky='w')
        self.detection_conf = tk.Scale(mp_frame, from_=0.1, to=1.0, resolution=0.1, orient=tk.HORIZONTAL,
                                     bg='#34495e', fg='white', length=200)
        self.detection_conf.set(0.7)
        self.detection_conf.grid(row=0, column=1, padx=10, pady=5)

        # Tracking confidence
        tk.Label(mp_frame, text="Tracking Confidence:", bg='#34495e', fg='white').grid(row=1, column=0, padx=10, pady=5, sticky='w')
        self.tracking_conf = tk.Scale(mp_frame, from_=0.1, to=1.0, resolution=0.1, orient=tk.HORIZONTAL,
                                    bg='#34495e', fg='white', length=200)
        self.tracking_conf.set(0.7)
        self.tracking_conf.grid(row=1, column=1, padx=10, pady=5)

        # Apply settings button
        tk.Button(mp_frame, text="Apply Settings", command=self.apply_settings,
                 bg='#2980b9', fg='white').grid(row=2, column=0, columnspan=2, pady=10)

    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180.0 else angle

    def extract_landmarks(self, results, frame_width, frame_height):
        """Extract and scale landmark coordinates"""
        if not results.pose_landmarks:
            return None

        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            visibility = landmark.visibility
            landmarks.append([x, y, visibility])

        return landmarks

    def update_analytics(self):
        """Update analytics charts"""
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()

        # Sample data for demonstration
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        reps_data = np.random.randint(10, 50, 30)  # Sample rep counts
        form_scores = np.random.uniform(0.7, 1.0, 30)  # Sample form scores

        # Plot 1: Progress over time
        self.ax1.plot(dates, reps_data, color='#3498db', linewidth=2)
        self.ax1.set_title('Repetitions Over Time', color='white', fontsize=12)
        self.ax1.set_xlabel('Date', color='white')
        self.ax1.set_ylabel('Repetitions', color='white')
        self.ax1.tick_params(colors='white')
        self.ax1.grid(True, alpha=0.3)

        # Plot 2: Form quality distribution
        self.ax2.hist(form_scores, bins=10, color='#e74c3c', alpha=0.7)
        self.ax2.set_title('Form Quality Distribution', color='white', fontsize=12)
        self.ax2.set_xlabel('Form Score', color='white')
        self.ax2.set_ylabel('Frequency', color='white')
        self.ax2.tick_params(colors='white')

        plt.tight_layout()
        self.canvas.draw()

    def export_data(self):
        """Export workout data to CSV"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )

            if filename:
                # Create sample data
                data = {
                    'date': pd.date_range(start='2024-01-01', periods=30, freq='D'),
                    'exercise': np.random.choice(['squat', 'pushup', 'lunge'], 30),
                    'repetitions': np.random.randint(10, 50, 30),
                    'form_score': np.random.uniform(0.7, 1.0, 30),
                    'duration_minutes': np.random.randint(5, 30, 30)
                }

                df = pd.DataFrame(data)
                df.to_csv(filename, index=False)
                messagebox.showinfo("Export Successful", f"Data exported to {filename}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")

    def retrain_models(self):
        """Retrain ML models with current data"""
        self.training_text.delete(1.0, tk.END)
        self.training_text.insert(tk.END, "Retraining models...\n\n")

        try:
            self.initialize_ml_models()
            self.training_text.insert(tk.END, "Models retrained successfully!\n")
            self.training_text.insert(tk.END, f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        except Exception as e:
            self.training_text.insert(tk.END, f"Training failed: {str(e)}\n")

    def load_training_data(self):
        """Load training data from file"""
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )

            if filename:
                with open(filename, 'r') as f:
                    data = json.load(f)
                    self.training_data = data

                messagebox.showinfo("Success", "Training data loaded successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load training data: {str(e)}")

    def save_models(self):
        """Save trained ML models"""
        try:
            for exercise in ['squat', 'pushup', 'lunge']:
                if self.ml_models[exercise] is not None:
                    # Save model
                    with open(f'{exercise}_model.pkl', 'wb') as f:
                        pickle.dump(self.ml_models[exercise], f)

                    # Save scaler
                    with open(f'{exercise}_scaler.pkl', 'wb') as f:
                        pickle.dump(self.scalers[exercise], f)

            messagebox.showinfo("Success", "Models saved successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save models: {str(e)}")

    def apply_settings(self):
        """Apply MediaPipe settings"""
        detection_conf = self.detection_conf.get()
        tracking_conf = self.tracking_conf.get()

        # Recreate pose estimator with new settings
        self.pose.close()
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
            static_image_mode=False
        )

        messagebox.showinfo("Settings Applied", 
                          f"Detection: {detection_conf}\nTracking: {tracking_conf}")

    def toggle_recording(self):
        """Toggle data recording for training"""
        self.is_recording = not self.is_recording

        if self.is_recording:
            self.record_button.config(text="STOP RECORDING", bg='#c0392b')
        else:
            self.record_button.config(text="RECORD DATA", bg='#f39c12')

    def start_workout(self):
        """Start workout session"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not access camera")
                return

            self.is_running = True
            self.counter = 0
            self.stage = "up"
            self.form_ok = True
            self.session_data = []

            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)

            # Start frame processing
            self.update_frame()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start workout: {str(e)}")

    def stop_workout(self):
        """Stop workout session"""
        self.is_running = False

        if self.cap:
            self.cap.release()
            self.cap = None

        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

        # Clear video display
        self.video_label.configure(image="", text="Video Feed\nPress Start to begin")
        self.video_label.image = None

        # Save session data
        if self.session_data:
            self.workout_history.append({
                'date': datetime.now(),
                'exercise': self.current_exercise,
                'reps': self.counter,
                'session_data': self.session_data
            })

        # Show summary
        messagebox.showinfo("Workout Complete", 
                          f"Exercise: {self.current_exercise.title()}\nTotal Reps: {self.counter}")

    def change_exercise(self):
        """Handle exercise change"""
        self.current_exercise = self.exercise_var.get()
        self.counter = 0
        self.stage = "up"

        if self.is_running:
            self.counter_label.configure(text="0")

    def update_frame(self):
        """Main frame processing loop with ML integration"""
        if not self.is_running or not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(15, self.update_frame)
            return

        # Flip frame and convert to RGB
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process pose estimation
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks:
            h, w = frame.shape[:2]
            landmarks = self.extract_landmarks(results, w, h)

            if landmarks:
                # Analyze form based on selected mode
                analysis_mode = self.analysis_mode.get()

                if analysis_mode == "ml":
                    ml_feedback, confidence = self.ml_analyze_form(landmarks)
                    feedback = [ml_feedback]
                    self.ml_confidence_label.config(text=f"ML Confidence: {confidence:.2f}")
                elif analysis_mode == "rule":
                    feedback = self.analyze_exercise_rules(landmarks)
                    self.ml_confidence_label.config(text="ML Confidence: N/A")
                else:  # hybrid
                    rule_feedback = self.analyze_exercise_rules(landmarks)
                    ml_feedback, confidence = self.ml_analyze_form(landmarks)
                    feedback = rule_feedback + [f"ML: {ml_feedback}"]
                    self.ml_confidence_label.config(text=f"ML Confidence: {confidence:.2f}")

                self.update_feedback_display(feedback)

                # Record data if enabled
                if self.is_recording:
                    features = self.extract_advanced_features(landmarks)
                    self.session_data.append({
                        'timestamp': time.time(),
                        'features': features.tolist(),
                        'stage': self.stage,
                        'counter': self.counter
                    })

                # Draw pose landmarks
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2)
                )

        # Add overlay information
        cv2.putText(frame, f'Exercise: {self.current_exercise.title()}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Reps: {self.counter}', (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Stage: {self.stage}', (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Mode: {self.analysis_mode.get().upper()}', (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Calculate FPS
        self.fps_counter += 1
        if time.time() - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = time.time()

        cv2.putText(frame, f'FPS: {self.current_fps}', (10, 190),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Convert and display frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((640, 480), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image=img)

        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo
        self.counter_label.configure(text=str(self.counter))
        self.fps_label.configure(text=f"FPS: {self.current_fps}")

        # Schedule next frame
        self.root.after(15, self.update_frame)

    def analyze_exercise_rules(self, landmarks):
        """Rule-based exercise analysis (simplified version)"""
        feedback = []

        try:
            if self.current_exercise == "squat":
                # Simplified squat analysis
                hip = np.mean([landmarks[23][:2], landmarks[24][:2]], axis=0)
                knee = np.mean([landmarks[25][:2], landmarks[26][:2]], axis=0)

                if self.stage == "up" and hip[1] > knee[1] + 20:
                    self.stage = "down"
                elif self.stage == "down" and hip[1] < knee[1] - 20:
                    self.counter += 1
                    self.stage = "up"
                    feedback.append("Good squat!")

            elif self.current_exercise == "pushup":
                # Simplified push-up analysis
                shoulder = np.mean([landmarks[11][:2], landmarks[12][:2]], axis=0)
                elbow = np.mean([landmarks[13][:2], landmarks[14][:2]], axis=0)

                elbow_angle = abs(shoulder[1] - elbow[1])

                if self.stage == "up" and elbow_angle > 30:
                    self.stage = "down"
                elif self.stage == "down" and elbow_angle < 10:
                    self.counter += 1
                    self.stage = "up"
                    feedback.append("Good push-up!")

        except Exception as e:
            feedback.append("Adjust your position")

        return feedback

    def update_feedback_display(self, feedback):
        """Update feedback display"""
        self.feedback_text.config(state=tk.NORMAL)
        self.feedback_text.delete(1.0, tk.END)

        timestamp = datetime.now().strftime("%H:%M:%S")
        self.feedback_text.insert(tk.END, f"[{timestamp}] {self.current_exercise.title()} Analysis\n\n")

        if feedback:
            for msg in feedback:
                self.feedback_text.insert(tk.END, f"• {msg}\n")
        else:
            self.feedback_text.insert(tk.END, "Keep going!\n")

        self.feedback_text.config(state=tk.DISABLED)
        self.feedback_text.see(tk.END)

    def on_closing(self):
        """Handle application closing"""
        if self.is_running:
            self.stop_workout()

        if self.pose:
            self.pose.close()

        self.root.destroy()

    def run(self):
        """Start the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

if __name__ == "__main__":
    print("Starting AI Workout Form Corrector - ML Enhanced Version...")
    app = MLWorkoutFormCorrector()
    app.run()
