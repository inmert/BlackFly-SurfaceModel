# webcam_demo.py
"""
Simple webcam demo for quick testing without complex dependencies
Works on all platforms (Windows, macOS, Linux)
"""

import cv2
import time
import argparse
import numpy as np
from pathlib import Path

class SimpleWebcamDetector:
    """Simple detector using only OpenCV (no ML dependencies)"""
    
    def __init__(self, detection_type='edges'):
        self.detection_type = detection_type
        
    def detect_edges(self, frame):
        """Simple edge detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        # Convert back to BGR for display
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
    def detect_motion(self, frame, prev_frame):
        """Simple motion detection"""
        if prev_frame is None:
            return frame
            
        # Calculate difference
        diff = cv2.absdiff(prev_frame, frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Threshold
        _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw motion areas
        output = frame.copy()
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
        return output
        
    def detect_faces(self, frame):
        """Simple face detection using OpenCV's built-in cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use OpenCV's pre-trained face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Draw rectangles
        output = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(output, "Face", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                       
        return output
        
    def process(self, frame, prev_frame=None):
        """Process frame based on detection type"""
        if self.detection_type == 'edges':
            return self.detect_edges(frame)
        elif self.detection_type == 'motion':
            return self.detect_motion(frame, prev_frame)
        elif self.detection_type == 'faces':
            return self.detect_faces(frame)
        else:
            return frame


class WebcamDemo:
    """Simple webcam demonstration app"""
    
    def __init__(self, camera_index=0, detection_type='edges'):
        self.camera_index = camera_index
        self.detector = SimpleWebcamDetector(detection_type)
        self.cap = None
        self.prev_frame = None
        self.recording = False
        self.video_writer = None
        
        # Create save directory
        self.save_path = Path('./webcam_captures')
        self.save_path.mkdir(exist_ok=True)
        
    def run(self):
        """Run the webcam demo"""
        # Open webcam with platform-specific backend
        import platform
        system = platform.system().lower()
        
        print(f"Starting webcam on {system}...")
        
        if system == 'darwin':  # macOS
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_AVFOUNDATION)
        elif system == 'windows':
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        else:  # Linux
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
            
        if not self.cap.isOpened():
            # Fallback to default
            self.cap = cv2.VideoCapture(self.camera_index)
            
        if not self.cap.isOpened():
            print(f"Error: Cannot open webcam {self.camera_index}")
            print("Try using --camera-index with a different number (0, 1, 2...)")
            return
            
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Get actual resolution
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Webcam opened: {width}x{height} @ {fps}fps")
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  'r' - Start/stop recording")
        print("  'd' - Switch detection mode")
        print("  'n' - No detection (normal view)")
        print("-" * 30)
        
        detection_modes = ['edges', 'motion', 'faces', 'none']
        mode_index = detection_modes.index(self.detector.detection_type)
        
        fps_time = time.time()
        fps_counter = 0
        fps_display = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error reading frame")
                    break
                    
                # Process frame
                if self.detector.detection_type == 'none':
                    output = frame
                else:
                    output = self.detector.process(frame, self.prev_frame)
                    
                # Calculate FPS
                fps_counter += 1
                current_time = time.time()
                if current_time - fps_time >= 1.0:
                    fps_display = fps_counter / (current_time - fps_time)
                    fps_counter = 0
                    fps_time = current_time
                    
                # Add text overlay
                cv2.putText(output, f"FPS: {fps_display:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(output, f"Mode: {self.detector.detection_type}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                           
                if self.recording:
                    cv2.circle(output, (width - 30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(output, "REC", (width - 80, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if self.video_writer:
                        self.video_writer.write(output)
                        
                # Display
                cv2.imshow('Webcam Demo', output)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = self.save_path / f"capture_{timestamp}.jpg"
                    cv2.imwrite(str(filename), output)
                    print(f"Saved: {filename}")
                elif key == ord('r'):
                    # Toggle recording
                    if not self.recording:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = self.save_path / f"recording_{timestamp}.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        self.video_writer = cv2.VideoWriter(str(filename), fourcc, 20.0, (width, height))
                        self.recording = True
                        print(f"Recording started: {filename}")
                    else:
                        self.recording = False
                        if self.video_writer:
                            self.video_writer.release()
                            self.video_writer = None
                        print("Recording stopped")
                elif key == ord('d'):
                    # Switch detection mode
                    mode_index = (mode_index + 1) % len(detection_modes)
                    self.detector.detection_type = detection_modes[mode_index]
                    print(f"Switched to: {self.detector.detection_type}")
                elif key == ord('n'):
                    # No detection
                    self.detector.detection_type = 'none'
                    print("Detection disabled")
                    
                # Store previous frame for motion detection
                self.prev_frame = frame.copy()
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            # Cleanup
            if self.recording and self.video_writer:
                self.video_writer.release()
            self.cap.release()
            cv2.destroyAllWindows()
            print("Webcam demo stopped")


def main():
    parser = argparse.ArgumentParser(description='Simple webcam demo with basic detection')
    parser.add_argument('--camera-index', type=int, default=0,
                       help='Camera index (0 for default webcam)')
    parser.add_argument('--detection', choices=['edges', 'motion', 'faces', 'none'],
                       default='faces', help='Detection type')
    
    args = parser.parse_args()
    
    demo = WebcamDemo(args.camera_index, args.detection)
    demo.run()


if __name__ == "__main__":
    main()