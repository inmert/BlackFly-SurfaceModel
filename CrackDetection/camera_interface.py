import PySpin
import numpy as np
from threading import Thread, Lock
import time
import cv2

class BlackflyCamera:
    """
    FLIR Blackfly GigE camera interface using Spinnaker SDK (PySpin)
    """
    
    def __init__(self, serial_number=None, width=1280, height=720, fps=30, exposure_time=10000):
        self.serial_number = serial_number
        self.width = width
        self.height = height
        self.fps = fps
        self.exposure_time = exposure_time  # microseconds
        
        self.system = None
        self.cam_list = None
        self.cam = None
        self.nodemap = None
        self.nodemap_tldevice = None
        
        self.frame = None
        self.frame_lock = Lock()
        self.running = False
        self.thread = None
        
    def start(self):
        """Initialize and start the camera"""
        print("Initializing Blackfly camera with PySpin...")
        
        # Retrieve singleton reference to system object
        self.system = PySpin.System.GetInstance()
        
        # Retrieve list of cameras from the system
        self.cam_list = self.system.GetCameras()
        
        num_cameras = self.cam_list.GetSize()
        print(f'Number of cameras detected: {num_cameras}')
        
        if num_cameras == 0:
            self.cam_list.Clear()
            self.system.ReleaseInstance()
            raise RuntimeError('No cameras detected!')
        
        # Select camera
        if self.serial_number:
            # Find camera by serial number
            self.cam = None
            for i in range(num_cameras):
                cam = self.cam_list.GetByIndex(i)
                nodemap_tldevice = cam.GetTLDeviceNodeMap()
                device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
                if PySpin.IsReadable(device_serial_number):
                    if device_serial_number.GetValue() == self.serial_number:
                        self.cam = cam
                        break
                cam.DeInit()
                del cam
            
            if self.cam is None:
                raise RuntimeError(f'Camera with serial number {self.serial_number} not found!')
        else:
            # Use first available camera
            self.cam = self.cam_list.GetByIndex(0)
        
        # Initialize camera
        self.cam.Init()
        
        # Retrieve GenICam nodemap
        self.nodemap = self.cam.GetNodeMap()
        self.nodemap_tldevice = self.cam.GetTLDeviceNodeMap()
        
        # Configure camera settings
        self._configure_camera()
        
        # Begin acquiring images
        self.cam.BeginAcquisition()
        
        # Print camera info
        self._print_camera_info()
        
        # Start capture thread
        self.running = True
        self.thread = Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        print("Camera started successfully!")
        
    def _configure_camera(self):
        """Configure camera settings"""
        try:
            # Set acquisition mode to continuous
            node_acquisition_mode = PySpin.CEnumerationPtr(self.nodemap.GetNode('AcquisitionMode'))
            if PySpin.IsWritable(node_acquisition_mode):
                node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
                acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
                node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
                print('Acquisition mode set to continuous...')
            
            # Set image dimensions
            node_width = PySpin.CIntegerPtr(self.nodemap.GetNode('Width'))
            if PySpin.IsWritable(node_width):
                node_width.SetValue(self.width)
                
            node_height = PySpin.CIntegerPtr(self.nodemap.GetNode('Height'))
            if PySpin.IsWritable(node_height):
                node_height.SetValue(self.height)
            
            # Set pixel format to BGR8 or Mono8
            node_pixel_format = PySpin.CEnumerationPtr(self.nodemap.GetNode('PixelFormat'))
            if PySpin.IsWritable(node_pixel_format):
                # Try BGR8 first, fallback to Mono8
                try:
                    node_pixel_format_bgr8 = node_pixel_format.GetEntryByName('BGR8')
                    pixel_format_bgr8 = node_pixel_format_bgr8.GetValue()
                    node_pixel_format.SetIntValue(pixel_format_bgr8)
                    print('Pixel format set to BGR8...')
                except:
                    node_pixel_format_mono8 = node_pixel_format.GetEntryByName('Mono8')
                    pixel_format_mono8 = node_pixel_format_mono8.GetValue()
                    node_pixel_format.SetIntValue(pixel_format_mono8)
                    print('Pixel format set to Mono8...')
            
            # Set frame rate
            node_acquisition_framerate_enable = PySpin.CBooleanPtr(self.nodemap.GetNode('AcquisitionFrameRateEnable'))
            if PySpin.IsWritable(node_acquisition_framerate_enable):
                node_acquisition_framerate_enable.SetValue(True)
                
            node_acquisition_framerate = PySpin.CFloatPtr(self.nodemap.GetNode('AcquisitionFrameRate'))
            if PySpin.IsWritable(node_acquisition_framerate):
                node_acquisition_framerate.SetValue(self.fps)
                print(f'Frame rate set to {self.fps} fps...')
            
            # Set exposure time
            node_exposure_auto = PySpin.CEnumerationPtr(self.nodemap.GetNode('ExposureAuto'))
            if PySpin.IsWritable(node_exposure_auto):
                node_exposure_auto_off = node_exposure_auto.GetEntryByName('Off')
                exposure_auto_off = node_exposure_auto_off.GetValue()
                node_exposure_auto.SetIntValue(exposure_auto_off)
                
            node_exposure_time = PySpin.CFloatPtr(self.nodemap.GetNode('ExposureTime'))
            if PySpin.IsWritable(node_exposure_time):
                node_exposure_time.SetValue(self.exposure_time)
                print(f'Exposure time set to {self.exposure_time} us...')
                
        except PySpin.SpinnakerException as ex:
            print(f'Error configuring camera: {ex}')
            
    def _capture_loop(self):
        """Continuous frame capture in separate thread"""
        while self.running:
            try:
                # Retrieve next received image
                image_result = self.cam.GetNextImage(1000)  # 1000ms timeout
                
                if image_result.IsIncomplete():
                    print(f'Image incomplete with image status {image_result.GetImageStatus()}')
                    image_result.Release()
                    continue
                
                # Convert image to numpy array
                image_data = image_result.GetNDArray()
                
                # Handle different pixel formats
                if len(image_data.shape) == 2:  # Mono8
                    # Convert to BGR for consistency
                    frame = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)
                else:  # BGR8 or other color formats
                    frame = image_data
                
                # Store frame thread-safely
                with self.frame_lock:
                    self.frame = frame.copy()
                
                image_result.Release()
                
            except PySpin.SpinnakerException as ex:
                print(f'Error capturing image: {ex}')
                time.sleep(0.01)
                
    def get_frame(self):
        """Get the latest frame (thread-safe)"""
        with self.frame_lock:
            return self.frame.copy() if self.frame is not None else None
            
    def stop(self):
        """Stop the camera and cleanup"""
        print("Stopping camera...")
        self.running = False
        
        if self.thread:
            self.thread.join()
            
        if self.cam:
            try:
                self.cam.EndAcquisition()
                self.cam.DeInit()
                del self.cam
            except:
                pass
                
        if self.cam_list:
            self.cam_list.Clear()
            
        if self.system:
            self.system.ReleaseInstance()
            
        print("Camera stopped")
        
    def _print_camera_info(self):
        """Print camera information"""
        try:
            device_serial_number = PySpin.CStringPtr(self.nodemap_tldevice.GetNode('DeviceSerialNumber'))
            if PySpin.IsReadable(device_serial_number):
                print(f'Camera Serial Number: {device_serial_number.GetValue()}')
                
            device_vendor_name = PySpin.CStringPtr(self.nodemap_tldevice.GetNode('DeviceVendorName'))
            if PySpin.IsReadable(device_vendor_name):
                print(f'Camera Vendor: {device_vendor_name.GetValue()}')
                
            device_model_name = PySpin.CStringPtr(self.nodemap_tldevice.GetNode('DeviceModelName'))
            if PySpin.IsReadable(device_model_name):
                print(f'Camera Model: {device_model_name.GetValue()}')
                
        except PySpin.SpinnakerException as ex:
            print(f'Error reading camera info: {ex}')
            
    def get_camera_info(self):
        """Get camera information as dictionary"""
        info = {}
        try:
            if self.nodemap_tldevice:
                device_serial_number = PySpin.CStringPtr(self.nodemap_tldevice.GetNode('DeviceSerialNumber'))
                if PySpin.IsReadable(device_serial_number):
                    info['serial_number'] = device_serial_number.GetValue()
                    
                device_model_name = PySpin.CStringPtr(self.nodemap_tldevice.GetNode('DeviceModelName'))
                if PySpin.IsReadable(device_model_name):
                    info['model'] = device_model_name.GetValue()
                    
            if self.nodemap:
                node_width = PySpin.CIntegerPtr(self.nodemap.GetNode('Width'))
                if PySpin.IsReadable(node_width):
                    info['width'] = node_width.GetValue()
                    
                node_height = PySpin.CIntegerPtr(self.nodemap.GetNode('Height'))
                if PySpin.IsReadable(node_height):
                    info['height'] = node_height.GetValue()
                    
                node_acquisition_framerate = PySpin.CFloatPtr(self.nodemap.GetNode('AcquisitionFrameRate'))
                if PySpin.IsReadable(node_acquisition_framerate):
                    info['fps'] = node_acquisition_framerate.GetValue()
                    
        except PySpin.SpinnakerException as ex:
            print(f'Error getting camera info: {ex}')
            
        return info
        
    @staticmethod
    def list_cameras():
        """List all available FLIR cameras"""
        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()
        
        print(f"Found {cam_list.GetSize()} FLIR camera(s):")
        
        cameras = []
        for i in range(cam_list.GetSize()):
            cam = cam_list.GetByIndex(i)
            nodemap_tldevice = cam.GetTLDeviceNodeMap()
            
            try:
                device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
                device_model_name = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceModelName'))
                
                serial = device_serial_number.GetValue() if PySpin.IsReadable(device_serial_number) else "Unknown"
                model = device_model_name.GetValue() if PySpin.IsReadable(device_model_name) else "Unknown"
                
                print(f"  Camera {i}: {model} (S/N: {serial})")
                cameras.append({'index': i, 'serial': serial, 'model': model})
                
            except PySpin.SpinnakerException as ex:
                print(f"  Camera {i}: Error reading info - {ex}")
                
            cam.DeInit()
            del cam
            
        cam_list.Clear()
        system.ReleaseInstance()
        
        return cameras