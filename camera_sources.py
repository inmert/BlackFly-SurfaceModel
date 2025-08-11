# camera_sources.py
"""
Camera source interfaces for Blackfly and Webcam
"""

import cv2
import numpy as np
from threading import Thread, Lock, Event
import time
from abc import ABC, abstractmethod
from config import CAMERA_CONFIG

# Helper class to manage a singleton PySpin instance
class _PySpinManager:
    """Manages a single instance of the PySpin System and camera list."""
    _system = None
    _cam_list = None
    _instance_lock = Lock()

    def __init__(self):
        # This class is not meant to be instantiated multiple times
        pass

    @classmethod
    def get_system(cls):
        with cls._instance_lock:
            if cls._system is None:
                try:
                    import PySpin
                    cls._system = PySpin.System.GetInstance()
                except ImportError:
                    raise RuntimeError("PySpin not installed. Install Spinnaker SDK and PySpin for Blackfly support.")
            return cls._system

    @classmethod
    def get_cam_list(cls):
        with cls._instance_lock:
            system = cls.get_system()
            if cls._cam_list is None:
                cls._cam_list = system.GetCameras()
            return cls._cam_list

    @classmethod
    def release_system(cls):
        with cls._instance_lock:
            if cls._cam_list is not None:
                cls._cam_list.Clear()
                cls._cam_list = None
            
            if cls._system is not None:
                # This check avoids an error if the system was never acquired
                if cls._system.IsInUse():
                    cls._system.ReleaseInstance()
                cls._system = None
            print("PySpin System released.")


class CameraSource(ABC):
    """Abstract base class for camera sources"""
    
    def __init__(self):
        self.frame = None
        self.frame_lock = Lock()
        self.running = False
        self.thread = None
        self.stop_event = Event()
        
    @abstractmethod
    def _initialize_camera(self):
        """Initialize the camera hardware"""
        pass
        
    @abstractmethod
    def _capture_frame(self):
        """Capture a single frame from camera"""
        pass
        
    @abstractmethod
    def _release_camera(self):
        """Release camera resources"""
        pass
        
    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        while self.running and not self.stop_event.is_set():
            try:
                frame = self._capture_frame()
                if frame is not None:
                    with self.frame_lock:
                        self.frame = frame.copy()
                else:
                    time.sleep(0.001)  # Small delay if no frame
            except Exception as e:
                print(f"Capture error: {e}")
                time.sleep(0.01)
                
    def start(self):
        """Start the camera capture"""
        try:
            self._initialize_camera()
            self.running = True
            self.stop_event.clear()
            self.thread = Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            print(f"{self.__class__.__name__} started successfully")
            return True
        except Exception as e:
            print(f"Failed to start {self.__class__.__name__}: {e}")
            return False
            
    def stop(self):
        """Stop the camera capture"""
        print(f"Stopping {self.__class__.__name__}...")
        self.running = False
        self.stop_event.set()
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            
        self._release_camera()
        print(f"{self.__class__.__name__} stopped")
        
    def get_frame(self):
        """Get the latest captured frame"""
        with self.frame_lock:
            return self.frame.copy() if self.frame is not None else None


class WebcamSource(CameraSource):
    """Webcam interface using OpenCV"""
    
    def __init__(self, camera_index=None, width=None, height=None, fps=None):
        super().__init__()
        config = CAMERA_CONFIG['webcam']
        self.camera_index = camera_index if camera_index is not None else config['camera_index']
        self.width = width if width is not None else config['width']
        self.height = height if height is not None else config['height']
        self.fps = fps if fps is not None else config['fps']
        self.cap = None
        
    def _initialize_camera(self):
        """Initialize webcam"""
        # Try different backends based on platform
        import platform
        system = platform.system().lower()
        
        if system == 'darwin':  # macOS
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_AVFOUNDATION)
        elif system == 'windows':
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        else:  # Linux
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
            
        if not self.cap.isOpened():
            # Fallback to default backend
            self.cap = cv2.VideoCapture(self.camera_index)
            
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open webcam {self.camera_index}")
            
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Read actual values (camera might not support requested values)
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Webcam initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
        
    def _capture_frame(self):
        """Capture frame from webcam"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None
        
    def _release_camera(self):
        """Release webcam"""
        if self.cap:
            self.cap.release()
            self.cap = None


class BlackflySource(CameraSource):
    """FLIR Blackfly camera interface"""
    
    def __init__(self, serial_number=None, width=None, height=None, fps=None, exposure_time=None):
        super().__init__()
        config = CAMERA_CONFIG['blackfly']
        self.serial_number = serial_number
        self.width = width if width is not None else config['width']
        self.height = height if height is not None else config['height'] 
        self.fps = fps if fps is not None else config['fps']
        self.exposure_time = exposure_time if exposure_time is not None else config['exposure_time']
        
        self.cam = None
        # PySpin is imported dynamically to avoid hard dependency
        self.pyspin_available = self._check_pyspin()

    def _check_pyspin(self):
        try:
            import PySpin
            return True
        except ImportError:
            return False

    def _initialize_camera(self):
        """Initialize Blackfly camera using the singleton manager"""
        if not self.pyspin_available:
            raise RuntimeError("PySpin not installed. Cannot initialize Blackfly camera.")

        cam_list = _PySpinManager.get_cam_list()
        
        if cam_list.GetSize() == 0:
            _PySpinManager.release_system()
            raise RuntimeError("No Blackfly cameras detected")
            
        # Select camera
        if self.serial_number:
            self.cam = self._find_camera_by_serial(self.serial_number, cam_list)
            if not self.cam:
                raise RuntimeError(f"Camera with serial {self.serial_number} not found")
        else:
            self.cam = cam_list.GetByIndex(0)
            
        # Initialize and configure
        self.cam.Init()
        self._configure_camera()
        self.cam.BeginAcquisition()
        
        print(f"Blackfly camera initialized: {self.width}x{self.height} @ {self.fps}fps")
        
    def _find_camera_by_serial(self, serial, cam_list):
        """Find camera by serial number without de-initializing others."""
        import PySpin
        for i in range(cam_list.GetSize()):
            cam = cam_list.GetByIndex(i)
            nodemap = cam.GetTLDeviceNodeMap()
            serial_node = PySpin.CStringPtr(nodemap.GetNode('DeviceSerialNumber'))
            if PySpin.IsReadable(serial_node) and serial_node.GetValue() == serial:
                return cam
        return None
        
    def _configure_camera(self):
        """Configure Blackfly camera settings"""
        import PySpin
        nodemap = self.cam.GetNodeMap()
        
        try:
            # Set continuous acquisition
            acq_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
            if PySpin.IsWritable(acq_mode):
                continuous = acq_mode.GetEntryByName('Continuous')
                acq_mode.SetIntValue(continuous.GetValue())
                
            # Set dimensions
            width_node = PySpin.CIntegerPtr(nodemap.GetNode('Width'))
            if PySpin.IsWritable(width_node):
                width_node.SetValue(self.width)
                
            height_node = PySpin.CIntegerPtr(nodemap.GetNode('Height'))
            if PySpin.IsWritable(height_node):
                height_node.SetValue(self.height)
                
            # Set pixel format
            pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
            if PySpin.IsWritable(pixel_format):
                try:
                    bgr8 = pixel_format.GetEntryByName('BGR8')
                    pixel_format.SetIntValue(bgr8.GetValue())
                except:
                    mono8 = pixel_format.GetEntryByName('Mono8')
                    pixel_format.SetIntValue(mono8.GetValue())
                    
            # Set frame rate
            fps_enable = PySpin.CBooleanPtr(nodemap.GetNode('AcquisitionFrameRateEnable'))
            if PySpin.IsWritable(fps_enable):
                fps_enable.SetValue(True)
                
            fps_node = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))
            if PySpin.IsWritable(fps_node):
                fps_node.SetValue(self.fps)
                
            # Set exposure
            exposure_auto = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
            if PySpin.IsWritable(exposure_auto):
                off = exposure_auto.GetEntryByName('Off')
                exposure_auto.SetIntValue(off.GetValue())
                
            exposure = PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))
            if PySpin.IsWritable(exposure):
                exposure.SetValue(self.exposure_time)
                
        except Exception as e:
            print(f"Warning: Could not configure all camera settings: {e}")
            
    def _capture_frame(self):
        """Capture frame from Blackfly"""
        if not self.cam:
            return None
            
        try:
            image_result = self.cam.GetNextImage(1000)
            
            if image_result.IsIncomplete():
                image_result.Release()
                return None
                
            # Convert to numpy array
            image_data = image_result.GetNDArray()
            
            # Convert mono to BGR if needed
            if len(image_data.shape) == 2:
                frame = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)
            else:
                frame = image_data
                
            image_result.Release()
            return frame
            
        except Exception:
            return None
            
    def _release_camera(self):
        """Release only the specific Blackfly camera instance."""
        if self.cam:
            try:
                if self.cam.IsStreaming():
                    self.cam.EndAcquisition()
                self.cam.DeInit()
            except Exception as e:
                print(f"Error releasing camera: {e}")
            self.cam = None

    @staticmethod
    def release_system():
        """Static method to release the global PySpin system."""
        _PySpinManager.release_system()
            
    @staticmethod
    def list_cameras():
        """List available Blackfly cameras using the singleton manager."""
        try:
            import PySpin
            cam_list = _PySpinManager.get_cam_list()
            
            num_cameras = cam_list.GetSize()
            print(f"Found {num_cameras} Blackfly camera(s):")
            cameras = []
            
            if num_cameras == 0:
                _PySpinManager.release_system()
                return []

            for i in range(num_cameras):
                cam = cam_list.GetByIndex(i)
                nodemap = cam.GetTLDeviceNodeMap()
                
                serial_node = PySpin.CStringPtr(nodemap.GetNode('DeviceSerialNumber'))
                model_node = PySpin.CStringPtr(nodemap.GetNode('DeviceModelName'))
                
                serial = serial_node.GetValue() if PySpin.IsReadable(serial_node) else "Unknown"
                model = model_node.GetValue() if PySpin.IsReadable(model_node) else "Unknown"
                
                print(f"  Camera {i}: {model} (S/N: {serial})")
                cameras.append({'index': i, 'serial': serial, 'model': model})

            # Do not release system here; main app will handle it.
            return cameras
            
        except ImportError:
            print("PySpin not installed. Cannot list Blackfly cameras.")
            return []
        except Exception as e:
            print(f"An error occurred while listing cameras: {e}")
            return []