# main.py
"""
Main application for real-time detection system
Supports both Blackfly cameras and webcams
"""

import cv2
import time
import argparse
import os
from pathlib import Path
from camera_sources import WebcamSource, BlackflySource
from base_detector import create_detector
from config import DISPLAY_CONFIG, SYSTEM_CONFIG

class DetectionApp:
    """Main detection application"""
    
    def __init__(self, camera_source, detector_name='crack_detection'):
        self.camera = camera_source
        self.detector = create_detector(detector_name)
        self.running = False
        self.fps = 0
        self.fps_counter = 0
        self.fps_time = time.time()
        
        # Create save directory if needed
        save_path = Path(SYSTEM_CONFIG.get('save_path', './detections'))
        save_path.mkdir(exist_ok=True)
        self.save_path = save_path
        
    def run(self):
        """Main application loop"""
        print("Starting detection system...")
        
        # Start camera
        if not self.camera.start():
            print("Failed to start camera")
            return
            
        # Give camera time to initialize
        time.sleep(1)
        
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Save current frame")
        print("  'r' - Reset detector")
        print("-" * 30)
        
        self.running = True
        
        try:
            while self.running:
                # Get frame
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue
                    
                # Process frame
                start_time = time.time()
                results = self.detector.process_frame(frame)
                processing_time = (time.time() - start_time) * 1000
                
                # Draw results
                output_frame = self.detector.draw_detections(frame, results)
                
                # Calculate FPS
                self._update_fps()
                
                # Add info overlay
                self._add_info_overlay(output_frame, processing_time, results)
                
                # Display
                cv2.imshow(DISPLAY_CONFIG['window_name'], output_frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_frame(output_frame)
                elif key == ord('r'):
                    print("Resetting detector...")
                    self.detector = create_detector(self.detector.__class__.__name__)
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            self.stop()
            
    def _update_fps(self):
        """Update FPS calculation"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_time >= 1.0:
            self.fps = self.fps_counter / (current_time - self.fps_time)
            self.fps_counter = 0
            self.fps_time = current_time
            
    def _add_info_overlay(self, frame, processing_time, results):
        """Add information overlay to frame"""
        # FPS and processing time
        info_text = f"FPS: {self.fps:.1f} | Process: {processing_time:.1f}ms"
        cv2.putText(frame, info_text, 
                   (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5,
                   DISPLAY_CONFIG['colors']['info'],
                   1)
        
        # Detection count
        detection_count = len(results.get('detections', []))
        if detection_count > 0:
            count_text = f"Detections: {detection_count}"
            cv2.putText(frame, count_text,
                       (10, frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       DISPLAY_CONFIG['colors']['detection'],
                       1)
                       
    def _save_frame(self, frame):
        """Save current frame with timestamp"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = self.save_path / f"detection_{timestamp}.jpg"
        cv2.imwrite(str(filename), frame)
        print(f"Saved: {filename}")
        
    def stop(self):
        """Stop the application"""
        print("\nStopping detection system...")
        self.running = False
        self.camera.stop()
        cv2.destroyAllWindows()
        print("Detection system stopped")


def main():
    parser = argparse.ArgumentParser(description='Real-time detection system')
    parser.add_argument('--source', choices=['webcam', 'blackfly'], default='webcam',
                       help='Camera source type')
    parser.add_argument('--detector', default='crack_detection',
                       choices=['crack_detection', 'object_detection', 
                               'aircraft_detection', 'segmentation'],
                       help='Detection model to use')
    parser.add_argument('--camera-index', type=int, default=0,
                       help='Webcam index (for webcam source)')
    parser.add_argument('--serial', type=str, default=None,
                       help='Blackfly camera serial number')
    parser.add_argument('--width', type=int, default=None,
                       help='Frame width')
    parser.add_argument('--height', type=int, default=None,
                       help='Frame height')
    parser.add_argument('--fps', type=int, default=None,
                       help='Frame rate')
    parser.add_argument('--exposure', type=int, default=None,
                       help='Exposure time (Blackfly only, microseconds)')
    parser.add_argument('--list-cameras', action='store_true',
                       help='List available Blackfly cameras')
    
    args = parser.parse_args()
    
    # List cameras if requested
    if args.list_cameras:
        BlackflySource.list_cameras()
        return
        
    # Create camera source
    if args.source == 'webcam':
        camera = WebcamSource(
            camera_index=args.camera_index,
            width=args.width,
            height=args.height,
            fps=args.fps
        )
    else:  # blackfly
        try:
            camera = BlackflySource(
                serial_number=args.serial,
                width=args.width,
                height=args.height,
                fps=args.fps,
                exposure_time=args.exposure
            )
        except RuntimeError as e:
            print(f"Error: {e}")
            print("Falling back to webcam...")
            camera = WebcamSource(camera_index=args.camera_index)
            
    # Create and run app
    app = DetectionApp(camera, args.detector)
    app.run()


if __name__ == "__main__":
    main()