# base_detector.py
"""
Modular detection system supporting multiple model types
"""

import cv2
import numpy as np
import torch
from PIL import Image
import warnings
import platform
from config import DETECTION_MODELS, SYSTEM_CONFIG

# Suppress warnings that can cause issues
warnings.filterwarnings('ignore')

class BaseDetector:
    """Base class for all detectors"""
    
    def __init__(self, model_config):
        self.config = model_config
        self.device = self._get_device()
        self.model = None
        self.processor = None
        
    def _get_device(self):
        """Get the appropriate device based on platform and availability"""
        if not SYSTEM_CONFIG.get('use_gpu', True):
            return 'cpu'
            
        system = platform.system().lower()
        
        if system == 'darwin':  # macOS
            # Check for M-series chip
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
            
        return 'cpu'
    
    def process_frame(self, frame):
        """Process a single frame - to be overridden by subclasses"""
        raise NotImplementedError
        
    def draw_detections(self, frame, results):
        """Draw detection results on frame"""
        from config import DISPLAY_CONFIG
        output = frame.copy()
        
        if results.get('detections'):
            for detection in results['detections']:
                if 'bbox' in detection:
                    # Bounding boxes from YOLO are [x_min, y_min, x_max, y_max]
                    # while our current code is expecting [x, y, w, h] from DETR
                    x1, y1, x2, y2 = detection['bbox']
                    cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), 
                                DISPLAY_CONFIG['colors']['bbox'], 2)
                    
                if 'label' in detection:
                    label = f"{detection['label']}: {detection.get('confidence', 0):.2f}"
                    cv2.putText(output, label, (int(x1), int(y1)-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              DISPLAY_CONFIG['font_scale'],
                              DISPLAY_CONFIG['colors']['text'],
                              DISPLAY_CONFIG['thickness'])
        
        # Add status text
        status = "DETECTIONS: " + str(len(results.get('detections', [])))
        cv2.putText(output, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   DISPLAY_CONFIG['font_scale'],
                   DISPLAY_CONFIG['colors']['info'],
                   DISPLAY_CONFIG['thickness'])
        
        return output


class CVDetector(BaseDetector):
    """Traditional computer vision detector (for cracks, edges, etc.)"""
    
    def process_frame(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            min_area = self.config.get('min_area', 50)
            aspect_ratio_threshold = self.config.get('aspect_ratio_threshold', 3)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                    # Draw contours on the cracks
                    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                    
                    if aspect_ratio > aspect_ratio_threshold:
                        detections.append({
                            'bbox': (x, y, w, h),
                            'confidence': min(area / 1000, 1.0),
                            'label': 'anomaly',
                            'contour': contour
                        })
            
            return {'detections': detections, 'success': True}
            
        except Exception as e:
            print(f"CV detection error: {e}")
            return {'detections': [], 'success': False, 'error': str(e)}


class ObjectDetector(BaseDetector):
    """Object detection using a configurable model framework"""
    
    def __init__(self, model_config):
        super().__init__(model_config)
        self._load_model()
        
    def _load_model(self):
        try:
            framework = self.config.get('framework', 'transformers')
            model_name = self.config['model_name']
            
            if framework == 'ultralytics':
                from torch.serialization import add_safe_globals
                from ultralytics.nn.tasks import DetectionModel
                from ultralytics.nn.modules import Conv, C2f, SPPF, Bottleneck, Concat, Detect, DFL
                import torch.nn as nn
                from torch.nn.modules.conv import Conv2d
                from torch.nn.modules.batchnorm import BatchNorm2d
                from torch.nn.modules.activation import SiLU
                from torch.nn.modules.container import ModuleList
                from torch.nn.modules.pooling import MaxPool2d
                from torch.nn.modules.upsampling import Upsample

                # Add all common YOLOv8 building blocks
                add_safe_globals([
                    DetectionModel,    # YOLO architecture
                    nn.Sequential,     # Sequential containers
                    Conv, C2f, SPPF, Bottleneck, Concat,Detect,DFL, # YOLO custom layers
                    Conv2d, BatchNorm2d, SiLU, 
                    ModuleList, MaxPool2d, Upsample     # Standard PyTorch layers
                ])

                from ultralytics import YOLO
                self.model = YOLO(model_name)
                print(f"Loaded YOLO model: {model_name} on {self.device}")


            elif framework == 'transformers':
                from transformers import pipeline
                device_map = 0 if self.device == 'cuda' else -1
                if self.device == 'mps':
                    device_map = -1  # Let transformers handle MPS
                self.model = pipeline(
                    "object-detection",
                    model=model_name,
                    device=device_map
                )
                print(f"Loaded transformer model: {model_name} on {self.device}")
            else:
                raise ValueError(f"Unsupported framework: {framework}")
                
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None
            
    def process_frame(self, frame):
        if self.model is None:
            return {'detections': [], 'success': False, 'error': 'Model not loaded'}
            
        try:
            framework = self.config.get('framework', 'transformers')
            confidence_threshold = self.config.get('confidence_threshold', 0.5)
            target_classes = self.config.get('target_classes')
            detections = []
            
            if framework == 'ultralytics':
                # Run YOLOv8 inference
                results = self.model(frame, conf=confidence_threshold, device=self.device, verbose=False)
                
                # Parse results
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls)
                        label = self.model.names[class_id]
                        
                        if target_classes and label not in target_classes:
                            continue
                        
                        confidence = float(box.conf)
                        # Bounding box coordinates in [x_min, y_min, x_max, y_max] format
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence,
                            'label': label
                        })
                        
            elif framework == 'transformers':
                # Convert to PIL Image for transformer models
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                predictions = self.model(pil_image)
                
                for pred in predictions:
                    if pred['score'] < confidence_threshold:
                        continue
                    if target_classes and pred['label'] not in target_classes:
                        continue
                        
                    box = pred['box']
                    # Bounding box coordinates in [x_min, y_min, x_max, y_max] format
                    x1, y1 = int(box['xmin']), int(box['ymin'])
                    x2, y2 = int(box['xmax']), int(box['ymax'])
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': pred['score'],
                        'label': pred['label']
                    })
                    
            return {'detections': detections, 'success': True}
            
        except Exception as e:
            print(f"Object detection error: {e}")
            return {'detections': [], 'success': False, 'error': str(e)}


class SegmentationDetector(BaseDetector):
    """Semantic/Instance segmentation detector"""
    
    def __init__(self, model_config):
        super().__init__(model_config)
        self._load_model()
        
    def _load_model(self):
        try:
            from transformers import pipeline
            
            device_map = 0 if self.device == 'cuda' else -1
            if self.device == 'mps':
                device_map = -1
                
            self.model = pipeline(
                "image-segmentation",
                model=self.config['model_name'],
                device=device_map
            )
            print(f"Loaded segmentation model: {self.config['model_name']}")
        except Exception as e:
            print(f"Failed to load segmentation model: {e}")
            self.model = None
            
    def process_frame(self, frame):
        if self.model is None:
            return {'detections': [], 'success': False, 'error': 'Model not loaded'}
            
        try:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            segments = self.model(pil_image)
            
            detections = []
            for i, segment in enumerate(segments):
                # Convert mask to contours
                mask = np.array(segment['mask'])
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    detections.append({
                        'bbox': (x, y, w, h),
                        'confidence': segment.get('score', 1.0),
                        'label': segment.get('label', f'segment_{i}'),
                        'mask': mask
                    })
                    
            return {'detections': detections, 'success': True}
            
        except Exception as e:
            print(f"Segmentation error: {e}")
            return {'detections': [], 'success': False, 'error': str(e)}


def create_detector(detector_name='crack_detection'):
    """Factory function to create appropriate detector"""
    if detector_name not in DETECTION_MODELS:
        print(f"Unknown detector: {detector_name}. Using crack_detection.")
        detector_name = 'crack_detection'
        
    config = DETECTION_MODELS[detector_name]
    detector_type = config.get('type', 'cv')
    
    if detector_type == 'cv':
        return CVDetector(config)
    elif detector_type == 'object_detection':
        return ObjectDetector(config)
    elif detector_type == 'segmentation':
        return SegmentationDetector(config)
    else:
        print(f"Unknown detector type: {detector_type}. Using CV detector.")
        return CVDetector(config)