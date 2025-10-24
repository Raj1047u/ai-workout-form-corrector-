# Create the complete AI Workout Form Corrector application
# This will be a comprehensive implementation based on the project specifications

# First, let's create the main application file
main_app_code = '''
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import time
import json
from datetime import datetime

class WorkoutFormCorrector:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize video capture
        self.cap = None
        
        # Exercise state variables
        self.current_exercise = "squat"
        self.stage = "up"
        self.counter = 0
        self.feedback = []
        self.form_ok = True
        
        # GUI variables
        self.is_running = False
        self.root = None
        self.video_label = None
        self.counter_label = None
        self.feedback_text = None
        self.exercise_var = None
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Initialize GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Initialize the main GUI window and components"""
        self.root = tk.Tk()
        self.root.title("AI Workout Form Corrector")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Video frame (left side)
        video_frame = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Video label
        self.video_label = tk.Label(video_frame, bg='black', text="Video Feed\\nPress Start to begin", 
                                  font=('Arial', 16), fg='white')
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control panel (right side)
        control_frame = tk.Frame(main_frame, bg='#34495e', width=300, relief=tk.RAISED, bd=2)
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
        
        # Control buttons
        button_frame = tk.Frame(control_frame, bg='#34495e')
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.start_button = tk.Button(button_frame, text="START", command=self.start_workout,
                                    bg='#27ae60', fg='white', font=('Arial', 12, 'bold'),
                                    height=2)
        self.start_button.pack(fill=tk.X, pady=2)
        
        self.stop_button = tk.Button(button_frame, text="STOP", command=self.stop_workout,
                                   bg='#e74c3c', fg='white', font=('Arial', 12, 'bold'),
                                   height=2, state=tk.DISABLED)
        self.stop_button.pack(fill=tk.X, pady=2)
        
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
        
        # FPS display
        self.fps_label = tk.Label(stats_frame, text="FPS: 0", bg='#34495e', fg='white', 
                                font=('Arial', 10))
        self.fps_label.pack(anchor=tk.W, padx=10, pady=(10, 5))
        
        # Feedback panel
        feedback_frame = tk.LabelFrame(control_frame, text="Real-time Feedback", 
                                     font=('Arial', 12, 'bold'), bg='#34495e', fg='white')
        feedback_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.feedback_text = tk.Text(feedback_frame, height=8, bg='#2c3e50', fg='white',
                                   font=('Arial', 10), wrap=tk.WORD, state=tk.DISABLED)
        self.feedback_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Status bar
        self.status_label = tk.Label(self.root, text="Ready to start workout", 
                                   bg='#34495e', fg='white', font=('Arial', 10))
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points using vector algebra"""
        a = np.array(a)  # First point
        b = np.array(b)  # Mid point (vertex)
        c = np.array(c)  # End point
        
        # Calculate vectors from vertex to other points
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
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
    
    def check_squat_form(self, landmarks):
        """Analyze squat form and provide feedback"""
        feedback = []
        self.form_ok = True
        
        try:
            # Key landmarks for squat analysis
            left_shoulder = landmarks[11][:2]
            right_shoulder = landmarks[12][:2]
            left_hip = landmarks[23][:2]
            right_hip = landmarks[24][:2]
            left_knee = landmarks[25][:2]
            right_knee = landmarks[26][:2]
            left_ankle = landmarks[27][:2]
            right_ankle = landmarks[28][:2]
            
            # Use average of left and right for stability
            shoulder = [(left_shoulder[0] + right_shoulder[0]) // 2, 
                       (left_shoulder[1] + right_shoulder[1]) // 2]
            hip = [(left_hip[0] + right_hip[0]) // 2, 
                   (left_hip[1] + right_hip[1]) // 2]
            knee = [(left_knee[0] + right_knee[0]) // 2, 
                    (left_knee[1] + right_knee[1]) // 2]
            ankle = [(left_ankle[0] + right_ankle[0]) // 2, 
                     (left_ankle[1] + right_ankle[1]) // 2]
            
            # Calculate hip angle (shoulder-hip-knee)
            hip_angle = self.calculate_angle(shoulder, hip, knee)
            
            # Calculate back angle (shoulder-hip-ankle) for posture
            back_angle = self.calculate_angle(shoulder, hip, ankle)
            
            # State machine for squat
            if self.stage == "up":
                if hip_angle < 120:  # Starting descent
                    self.stage = "down"
                    self.form_ok = True
            elif self.stage == "down":
                # Check squat depth
                if hip_angle < 100:  # Good depth achieved
                    if hip_angle > 160:  # Coming back up
                        if self.form_ok:
                            self.counter += 1
                            feedback.append("Good rep! ✓")
                        self.stage = "up"
                else:
                    if hip_angle > 140:  # Coming up without proper depth
                        feedback.append("Go deeper!")
                        self.form_ok = False
                        self.stage = "up"
                
                # Check back posture
                if back_angle < 80:
                    feedback.append("Keep chest up, back straight!")
                    self.form_ok = False
                    
        except (IndexError, ValueError) as e:
            feedback.append("Position yourself in camera view")
            
        return feedback
    
    def check_pushup_form(self, landmarks):
        """Analyze push-up form and provide feedback"""
        feedback = []
        self.form_ok = True
        
        try:
            # Key landmarks for push-up analysis
            left_shoulder = landmarks[11][:2]
            right_shoulder = landmarks[12][:2]
            left_elbow = landmarks[13][:2]
            right_elbow = landmarks[14][:2]
            left_wrist = landmarks[15][:2]
            right_wrist = landmarks[16][:2]
            left_hip = landmarks[23][:2]
            right_hip = landmarks[24][:2]
            left_ankle = landmarks[27][:2]
            right_ankle = landmarks[28][:2]
            
            # Use left side for primary analysis
            shoulder = left_shoulder
            elbow = left_elbow
            wrist = left_wrist
            hip = [(left_hip[0] + right_hip[0]) // 2, 
                   (left_hip[1] + right_hip[1]) // 2]
            ankle = [(left_ankle[0] + right_ankle[0]) // 2, 
                     (left_ankle[1] + right_ankle[1]) // 2]
            
            # Calculate elbow angle
            elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
            
            # Calculate body alignment (shoulder-hip-ankle)
            body_angle = self.calculate_angle(shoulder, hip, ankle)
            
            # State machine for push-up
            if self.stage == "up":
                if elbow_angle < 140:  # Starting descent
                    self.stage = "down"
                    self.form_ok = True
            elif self.stage == "down":
                # Check range of motion
                if elbow_angle < 95:  # Good range achieved
                    if elbow_angle > 160:  # Coming back up
                        if self.form_ok:
                            self.counter += 1
                            feedback.append("Good rep! ✓")
                        self.stage = "up"
                else:
                    if elbow_angle > 140:  # Coming up without proper range
                        feedback.append("Lower your chest!")
                        self.form_ok = False
                        self.stage = "up"
                
                # Check body alignment
                if body_angle < 160:
                    feedback.append("Keep body straight!")
                    self.form_ok = False
                    
        except (IndexError, ValueError) as e:
            feedback.append("Position yourself in camera view")
            
        return feedback
    
    def check_lunge_form(self, landmarks):
        """Analyze lunge form and provide feedback"""
        feedback = []
        self.form_ok = True
        
        try:
            # Key landmarks for lunge analysis
            left_hip = landmarks[23][:2]
            right_hip = landmarks[24][:2]
            left_knee = landmarks[25][:2]
            right_knee = landmarks[26][:2]
            left_ankle = landmarks[27][:2]
            right_ankle = landmarks[28][:2]
            
            # Determine which leg is forward (lower y-coordinate)
            if left_knee[1] < right_knee[1]:  # Left leg forward
                front_hip, front_knee, front_ankle = left_hip, left_knee, left_ankle
                back_hip, back_knee, back_ankle = right_hip, right_knee, right_ankle
            else:  # Right leg forward
                front_hip, front_knee, front_ankle = right_hip, right_knee, right_ankle
                back_hip, back_knee, back_ankle = left_hip, left_knee, left_ankle
            
            # Calculate knee angles
            front_knee_angle = self.calculate_angle(front_hip, front_knee, front_ankle)
            back_knee_angle = self.calculate_angle(back_hip, back_knee, back_ankle)
            
            # State machine for lunge
            if self.stage == "up":
                if front_knee_angle < 140 and back_knee_angle < 140:  # Starting descent
                    self.stage = "down"
                    self.form_ok = True
            elif self.stage == "down":
                # Check proper depth
                if front_knee_angle < 110 and back_knee_angle < 110:  # Good depth
                    if front_knee_angle > 150 and back_knee_angle > 150:  # Coming up
                        if self.form_ok:
                            self.counter += 1
                            feedback.append("Good rep! ✓")
                        self.stage = "up"
                else:
                    if front_knee_angle > 130 or back_knee_angle > 130:  # Coming up without depth
                        feedback.append("Bend knees to 90 degrees!")
                        self.form_ok = False
                        self.stage = "up"
                
                # Check front knee position (shouldn't extend too far forward)
                if front_knee[0] > front_ankle[0] + 50:  # Knee too far forward
                    feedback.append("Keep front knee behind toes!")
                    self.form_ok = False
                    
        except (IndexError, ValueError) as e:
            feedback.append("Position yourself in camera view")
            
        return feedback
    
    def analyze_exercise(self, landmarks):
        """Route to appropriate exercise analysis"""
        if self.current_exercise == "squat":
            return self.check_squat_form(landmarks)
        elif self.current_exercise == "pushup":
            return self.check_pushup_form(landmarks)
        elif self.current_exercise == "lunge":
            return self.check_lunge_form(landmarks)
        else:
            return ["Select an exercise to begin"]
    
    def update_feedback_display(self, feedback):
        """Update the feedback text widget"""
        self.feedback_text.config(state=tk.NORMAL)
        self.feedback_text.delete(1.0, tk.END)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.feedback_text.insert(tk.END, f"[{timestamp}] {self.current_exercise.title()}\\n\\n")
        
        if feedback:
            for msg in feedback:
                color = "green" if "✓" in msg or "Good" in msg else "yellow"
                self.feedback_text.insert(tk.END, f"• {msg}\\n")
        else:
            self.feedback_text.insert(tk.END, "Form looks good!\\n")
            
        self.feedback_text.config(state=tk.DISABLED)
        self.feedback_text.see(tk.END)
    
    def update_frame(self):
        """Main frame processing loop"""
        if not self.is_running or not self.cap or not self.cap.isOpened():
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(15, self.update_frame)
            return
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process pose estimation
        results = self.pose.process(frame_rgb)
        
        # Extract landmarks and analyze form
        if results.pose_landmarks:
            h, w = frame.shape[:2]
            landmarks = self.extract_landmarks(results, w, h)
            
            if landmarks:
                feedback = self.analyze_exercise(landmarks)
                self.update_feedback_display(feedback)
                
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
        
        # Calculate and display FPS
        self.fps_counter += 1
        if time.time() - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = time.time()
            
        cv2.putText(frame, f'FPS: {self.current_fps}', (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Convert frame for Tkinter display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((640, 480), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image=img)
        
        # Update GUI
        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo
        self.counter_label.configure(text=str(self.counter))
        self.fps_label.configure(text=f"FPS: {self.current_fps}")
        
        # Schedule next frame
        self.root.after(15, self.update_frame)
    
    def start_workout(self):
        """Start the workout session"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not access camera")
                return
            
            self.is_running = True
            self.counter = 0
            self.stage = "up"
            self.form_ok = True
            
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text=f"Workout started - {self.current_exercise.title()}")
            
            # Start frame processing
            self.update_frame()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start workout: {str(e)}")
    
    def stop_workout(self):
        """Stop the workout session"""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Workout stopped")
        
        # Clear video display
        self.video_label.configure(image="", text="Video Feed\\nPress Start to begin")
        self.video_label.image = None
        
        # Show workout summary
        summary_msg = f"Workout Summary\\n\\nExercise: {self.current_exercise.title()}\\nTotal Reps: {self.counter}"
        messagebox.showinfo("Workout Complete", summary_msg)
    
    def change_exercise(self):
        """Handle exercise selection change"""
        self.current_exercise = self.exercise_var.get()
        self.counter = 0
        self.stage = "up"
        self.status_label.config(text=f"Exercise changed to: {self.current_exercise.title()}")
        
        if self.is_running:
            self.counter_label.configure(text="0")
    
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
    app = WorkoutFormCorrector()
    app.run()
'''

# Save the main application file
with open("ai_workout_form_corrector.py", "w") as f:
    f.write(main_app_code)

print("✅ Created main application file: ai_workout_form_corrector.py")