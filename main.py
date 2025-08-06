import cv2
import time
import argparse
from camera_interface import BlackflyCamera
from model_interface import CrackDetector

class RealTimeCrackDetection:
    """
    Main application class that combines camera and detection
    """
    
    def __init__(self, serial_number=None, model_name="microsoft/DiNAT-Large-ImageNet-1K-224", 
                 width=1280, height=720, fps=30, exposure_time=10000):
        self.camera = BlackflyCamera(
            serial_number=serial_number,
            width=width,
            height=height, 
            fps=fps,
            exposure_time=exposure_time
        )
        self.detector = CrackDetector(model_name=model_name)
        self.fps_counter = 0
        self.fps_time = time.time()
        self.running = False
        
    def start(self):
        """Start the real-time detection system"""
        print("Starting real-time crack detection system...")
        
        # Initialize camera
        self.camera.start()
        time.sleep(2)  # Give camera time to warm up
        
        # Print camera info
        cam_info = self.camera.get_camera_info()
        if cam_info:
            print(f"Camera info: {cam_info}")
        
        self.running = True
        self.run_detection_loop()
        
    def run_detection_loop(self):
        """Main detection loop"""
        print("Detection loop started. Press 'q' to quit, 's' to save current frame.")
        
        while self.running:
            # Get frame from camera
            frame = self.camera.get_frame()
            if frame is None:
                continue
                
            # Process frame for anomalies
            start_time = time.time()
            results = self.detector.process_frame(frame)
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Draw detections
            output_frame = self.detector.draw_detections(frame, results)
            
            # Add FPS and processing time info
            self.fps_counter += 1
            current_time = time.time()
            if current_time - self.fps_time >= 1.0:
                fps = self.fps_counter / (current_time - self.fps_time)
                self.fps_counter = 0
                self.fps_time = current_time
            else:
                fps = 0
                
            if fps > 0:
                info_text = f"FPS: {fps:.1f} | Processing: {processing_time:.1f}ms"
                cv2.putText(output_frame, info_text, (10, output_frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the result
            cv2.imshow('Real-time Crack Detection', output_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame with detections
                timestamp = int(time.time())
                filename = f"crack_detection_{timestamp}.jpg"
                cv2.imwrite(filename, output_frame)
                print(f"Saved frame: {filename}")
                
        self.stop()
        
    def stop(self):
        """Stop the detection system"""
        print("Stopping detection system...")
        self.running = False
        self.camera.stop()
        cv2.destroyAllWindows()
        print("Detection system stopped.")

def main():
    parser = argparse.ArgumentParser(description='Real-time crack detection using FLIR Blackfly camera')
    parser.add_argument('--serial', type=str, default=None, 
                       help='Camera serial number (uses first available if not specified)')
    parser.add_argument('--model', type=str, default="microsoft/DiNAT-Large-ImageNet-1K-224", 
                       help='Hugging Face model name')
    parser.add_argument('--width', type=int, default=1280, help='Image width')
    parser.add_argument('--height', type=int, default=720, help='Image height')
    parser.add_argument('--fps', type=int, default=30, help='Frame rate')
    parser.add_argument('--exposure', type=int, default=10000, help='Exposure time in microseconds')
    parser.add_argument('--list-cameras', action='store_true', help='List available cameras and exit')
    
    args = parser.parse_args()
    
    # List cameras if requested
    if args.list_cameras:
        BlackflyCamera.list_cameras()
        return
    
    try:
        app = RealTimeCrackDetection(
            serial_number=args.serial,
            model_name=args.model,
            width=args.width,
            height=args.height,
            fps=args.fps,
            exposure_time=args.exposure
        )
        app.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        print("\nTry running with --list-cameras to see available cameras")

if __name__ == "__main__":
    main()