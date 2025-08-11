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
                    x, y, w, h = detection['bbox']
                    cv2.rectangle(output, (x, y), (x+w, y+h), 
                                DISPLAY_CONFIG['colors']['bbox'], 2)
                    
                if 'label' in detection:
                    label = f"{detection['label']}: {detection.get('confidence', 0):.2f}"
                    cv2.putText(output, label, (x, y-10),
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
    """Object detection using transformer models"""
    
    def __init__(self, model_config):
        super().__init__(model_config)
        self._load_model()
        
    def _load_model(self):
        try:
            from transformers import pipeline
            
            device_map = 0 if self.device == 'cuda' else -1
            if self.device == 'mps':
                device_map = -1  # Let transformers handle MPS
                
            self.model = pipeline(
                "object-detection",
                model=self.config['model_name'],
                device=device_map
            )
            print(f"Loaded model: {self.config['model_name']} on {self.device}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None
            
    def process_frame(self, frame):
        if self.model is None:
            return {'detections': [], 'success': False, 'error': 'Model not loaded'}
            
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Run detection
            predictions = self.model(pil_image)
            
            detections = []
            target_classes = self.config.get('target_classes')
            confidence_threshold = self.config.get('confidence_threshold', 0.5)
            
            for pred in predictions:
                if pred['score'] < confidence_threshold:
                    continue
                    
                if target_classes and pred['label'] not in target_classes:
                    continue
                    
                # Convert to opencv coordinates
                box = pred['box']
                x, y = int(box['xmin']), int(box['ymin'])
                w, h = int(box['xmax'] - box['xmin']), int(box['ymax'] - box['ymin'])
                
                detections.append({
                    'bbox': (x, y, w, h),
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