# config.py
"""
Configuration settings for the detection system
"""

# Camera Settings
CAMERA_CONFIG = {
    'blackfly': {
        'width': 1280,
        'height': 720,
        'fps': 30,
        'exposure_time': 10000,  # microseconds
    },
    'webcam': {
        'width': 1280,
        'height': 720,
        'fps': 30,
        'camera_index': 0,  # Default webcam
    }
}

# Model Configurations
DETECTION_MODELS = {
    'crack_detection': {
        'type': 'cv',  # 'cv' for computer vision, 'classification', 'object_detection', 'segmentation'
        'model_name': None,  # Using CV methods for cracks
        'confidence_threshold': 0.3,
        'min_area': 50,
        'aspect_ratio_threshold': 3,
    },
    'detr_object_detection': {  # Renamed for clarity
        'type': 'object_detection',
        'model_name': 'facebook/detr-resnet-50',  # DETR for object detection
        'confidence_threshold': 0.5,
        'target_classes': None,  # None means all classes
    },
    'yolov8_aircraft_detection': {
        'type': 'object_detection',
        'model_name': 'yolov8n.pt',  # YOLOv8 nano model for high speed
        'confidence_threshold': 0.2,
        'target_classes': ['airplane'], # Or other class names from the dataset if needed
        'framework': 'ultralytics', # Specify the framework for loading
    },
    'segmentation': {
        'type': 'segmentation',
        'model_name': 'facebook/mask2former-swin-base-coco-panoptic',
        'confidence_threshold': 0.5,
    }
}

# Display Settings
DISPLAY_CONFIG = {
    'window_name': 'Real-time Detection',
    'font_scale': 0.7,
    'thickness': 2,
    'colors': {
        'detection': (0, 0, 255),  # Red in BGR
        'bbox': (255, 0, 0),       # Blue in BGR
        'text': (0, 255, 0),        # Green in BGR
        'info': (255, 255, 255),   # White in BGR
    }
}

# System Settings
SYSTEM_CONFIG = {
    'use_gpu': True,  # Will fallback to CPU if CUDA not available
    'log_level': 'INFO',
}

# Platform-specific settings
import platform
PLATFORM = platform.system().lower()

if PLATFORM == 'darwin':  
    # M-series Macs use MPS (Metal Performance Shaders)
    SYSTEM_CONFIG['device_preference'] = 'mps'
elif PLATFORM == 'windows':
    # CUDA is preferred on Windows
    SYSTEM_CONFIG['device_preference'] = 'cuda'
else:  
    # Linux
    SYSTEM_CONFIG['device_preference'] = 'cuda'