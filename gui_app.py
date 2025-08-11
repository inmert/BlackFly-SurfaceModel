# gui_app.py
"""
PyQt6-based graphical user interface for the real-time detection system.
This file integrates the existing camera and detection logic into a responsive GUI.
"""

import sys
import cv2
import numpy as np
import time
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QLabel, QPushButton, QComboBox, QSizePolicy
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QMutex

# Import the core logic from the existing project files
from camera_sources import WebcamSource, BlackflySource
from base_detector import create_detector
from config import DETECTION_MODELS, CAMERA_CONFIG, DISPLAY_CONFIG

# Define a Worker Thread to handle camera and detection processing
# This prevents the GUI from freezing during heavy computation
class WorkerThread(QThread):
    # Signals to communicate with the GUI thread
    frame_ready = pyqtSignal(QPixmap)
    status_ready = pyqtSignal(str)
    
    def __init__(self, camera_source, detector_name):
        super().__init__()
        self._camera_source = camera_source
        self._detector_name = detector_name
        self.running = True
        self.mutex = QMutex()
        
    def run(self):
        """
        Main loop for the worker thread. Grabs frames, processes them,
        and emits signals to update the GUI.
        """
        print("Worker thread started.")
        self.camera = self._camera_source
        self.detector = create_detector(self._detector_name)
        
        # Start camera capture
        if not self.camera.start():
            self.status_ready.emit("Failed to start camera")
            self.running = False
            return
            
        # Give camera time to initialize
        time.sleep(1)

        fps_counter = 0
        fps_time = time.time()
        fps = 0.0
        
        try:
            while self.running:
                # Get frame from the camera source
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue
                
                # Perform detection on the frame
                start_time = time.time()
                results = self.detector.process_frame(frame)
                processing_time = (time.time() - start_time) * 1000
                
                # Draw detections on the frame
                output_frame = self.detector.draw_detections(frame, results)
                
                # Correct the color channels from BGR (OpenCV) to RGB (PyQt)
                rgb_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                
                # Convert the RGB frame to a QPixmap for display
                height, width, channel = rgb_frame.shape
                bytes_per_line = 3 * width
                q_image = QImage(
                    rgb_frame.data,
                    width,
                    height,
                    bytes_per_line,
                    QImage.Format.Format_RGB888
                )
                pixmap = QPixmap.fromImage(q_image)
                
                # Emit the processed frame to the GUI
                self.frame_ready.emit(pixmap)
                
                # Update FPS calculation
                fps_counter += 1
                current_time = time.time()
                if current_time - fps_time >= 1.0:
                    fps = fps_counter / (current_time - fps_time)
                    fps_counter = 0
                    fps_time = current_time
                    
                # Emit status information to the GUI
                detection_count = len(results.get('detections', []))
                status_text = f"FPS: {fps:.1f} | Process: {processing_time:.1f}ms | Detections: {detection_count}"
                    
                # Small delay to prevent busy-looping
                time.sleep(0.001)
                
        except Exception as e:
            print(f"Worker thread error: {e}")
        finally:
            self.camera.stop()
            print("Worker thread stopped.")

    def stop(self):
        """Method to gracefully stop the worker thread."""
        self.mutex.lock()
        self.running = False
        self.mutex.unlock()
        self.wait() # Wait for the thread to finish

    def reset_detector(self, detector_name):
        """Resets the detector instance."""
        print(f"Resetting detector to {detector_name} from worker thread...")
        self.detector = create_detector(detector_name)


class DetectionAppGUI(QMainWindow):
    """
    Main PyQt6 GUI application window.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Detection System")
        self.setGeometry(100, 100, 1000, 800)

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        self.worker = None
        self.camera_source = None
        self.is_running = False

        self._setup_ui()
    
    def _setup_ui(self):
        """Initializes the user interface components."""
        
        # Video display area
        self.video_label = QLabel("Video Feed")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #333; color: white;")
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.main_layout.addWidget(self.video_label, 1)

        # Status Bar
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("font-weight: bold; padding: 5px;")
        self.main_layout.addWidget(self.status_label)

        # Controls layout
        controls_layout = QHBoxLayout()
        self.main_layout.addLayout(controls_layout)

        # Camera selection dropdown
        controls_layout.addWidget(QLabel("Camera Source:"))
        self.camera_selector = QComboBox()
        self.camera_selector.addItems(["Webcam", "Blackfly"])
        self.camera_selector.setCurrentIndex(CAMERA_CONFIG['webcam']['camera_index'])
        controls_layout.addWidget(self.camera_selector)
        
        # Detector selection dropdown
        controls_layout.addWidget(QLabel("Detector Model:"))
        self.detector_selector = QComboBox()
        self.detector_selector.addItems(DETECTION_MODELS.keys())
        controls_layout.addWidget(self.detector_selector)

        # Control buttons
        self.start_button = QPushButton("Start Detection")
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; font-weight: bold;")
        self.start_button.clicked.connect(self.start_detection)
        controls_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Detection")
        self.stop_button.setStyleSheet("background-color: #F44336; color: white; padding: 10px; font-weight: bold;")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_detection)
        controls_layout.addWidget(self.stop_button)

        self.reset_button = QPushButton("Reset Detector")
        self.reset_button.setStyleSheet("background-color: #FFC107; color: black; padding: 10px; font-weight: bold;")
        self.reset_button.setEnabled(False)
        self.reset_button.clicked.connect(self.reset_detector)
        controls_layout.addWidget(self.reset_button)

    def start_detection(self):
        """Initializes and starts the worker thread for detection."""
        if self.is_running:
            return

        camera_type = self.camera_selector.currentText()
        detector_name = self.detector_selector.currentText()
        
        # Create the selected camera source with better error handling for Blackfly
        self.camera_source = None
        if camera_type == "Webcam":
            self.camera_source = WebcamSource()
        else: # Blackfly
            try:
                # Check for available Blackfly cameras first to avoid a blocking call
                if not BlackflySource.list_cameras():
                    self.status_label.setText("Error: No Blackfly camera found. Please connect a camera.")
                    self.stop_detection() # Revert UI state
                    return
                self.camera_source = BlackflySource()
            except RuntimeError as e:
                print(f"Error: {e}")
                self.status_label.setText(f"Error: {e}")
                self.stop_detection() # Revert UI state
                return
            finally:
                # Always release the system after checking for cameras
                BlackflySource.release_system()


        if not self.camera_source:
             self.status_label.setText("Error: Camera source not initialized.")
             return
        
        # Create and start the worker thread
        self.worker = WorkerThread(self.camera_source, detector_name)
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.status_ready.connect(self.update_status)
        self.worker.start()
        
        # Update GUI state
        self.is_running = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.camera_selector.setEnabled(False)
        self.detector_selector.setEnabled(False)

    def stop_detection(self):
        """Stops the worker thread and cleans up resources."""
        if not self.is_running:
            return
            
        if self.worker:
            self.worker.stop()
            self.worker = None

        self.video_label.clear()
        self.video_label.setText("Video Feed")

        # Update GUI state
        self.is_running = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.camera_selector.setEnabled(True)
        self.detector_selector.setEnabled(True)
        self.status_label.setText("Status: Stopped")

    def reset_detector(self):
        """Resets the detector model in the worker thread."""
        if self.is_running and self.worker:
            detector_name = self.detector_selector.currentText()
            self.worker.reset_detector(detector_name)
            self.status_label.setText(f"Status: Detector reset to {detector_name}")

    def update_frame(self, pixmap):
        """
        Slot to receive a QPixmap from the worker thread and display it.
        Resizes the pixmap to fit the label.
        """
        if self.is_running:
            self.video_label.setPixmap(pixmap.scaled(
                self.video_label.width(),
                self.video_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio
            ))
            
    def update_status(self, text):
        """
        Slot to receive status text from the worker thread.
        """
        self.status_label.setText(f"Status: {text}")

    def closeEvent(self, event):
        """
        Gracefully stop the worker thread when the main window is closed.
        """
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DetectionAppGUI()
    window.show()
    sys.exit(app.exec())
