import cv2
import numpy as np
from transformers import pipeline
import torch
from PIL import Image
from abc import ABC, abstractmethod

class BaseVisionProcessor(ABC):
    """Base class for all vision processing models"""
    
    def __init__(self, model_name, confidence_threshold=0.3, device=None):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Hugging Face pipeline"""
        try:
            device_id = 0 if self.device == "cuda" else -1
            self.pipeline = pipeline(
                self.get_task_type(),
                model=self.model_name,
                device=device_id
            )
            print(f"Loaded {self.get_task_type()} model: {self.model_name}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            self.pipeline = None
    
    @abstractmethod
    def get_task_type(self):
        """Return the Hugging Face task type (e.g., 'object-detection')"""
        pass
    
    @abstractmethod
    def process_frame(self, frame):
        """Process a single frame and return results"""
        pass
    
    @abstractmethod
    def draw_results(self, frame, results):
        """Draw results on the frame"""
        pass
    
    def is_available(self):
        """Check if model is loaded and ready"""
        return self.pipeline is not None


class ObjectDetectionProcessor(BaseVisionProcessor):
    """Object detection using DETR, YOLO, or similar models"""
    
    def get_task_type(self):
        return "object-detection"
    
    def process_frame(self, frame):
        """Process frame for object detection"""
        if not self.is_available():
            return {'detections': [], 'count': 0}
        
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Run detection
            detections = self.pipeline(pil_image)
            
            # Filter by confidence threshold
            filtered_detections = [
                det for det in detections 
                if det['score'] >= self.confidence_threshold
            ]
            
            return {
                'detections': filtered_detections,
                'count': len(filtered_detections),
                'raw_detections': detections
            }
            
        except Exception as e:
            print(f"Object detection error: {e}")
            return {'detections': [], 'count': 0}
    
    def draw_results(self, frame, results):
        """Draw bounding boxes and labels"""
        output_frame = frame.copy()
        
        for detection in results['detections']:
            # Get bounding box coordinates
            box = detection['box']
            x1, y1 = int(box['xmin']), int(box['ymin'])
            x2, y2 = int(box['xmax']), int(box['ymax'])
            
            # Draw bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label and confidence
            label = f"{detection['label']}: {detection['score']:.2f}"
            cv2.putText(output_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add count info
        count_text = f"Objects detected: {results['count']}"
        cv2.putText(output_frame, count_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return output_frame


class ImageClassificationProcessor(BaseVisionProcessor):
    """Image classification for scene analysis or anomaly detection"""
    
    def get_task_type(self):
        return "image-classification"
    
    def process_frame(self, frame):
        """Process frame for classification"""
        if not self.is_available():
            return {'predictions': [], 'top_class': None, 'confidence': 0.0}
        
        try:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            predictions = self.pipeline(pil_image)
            
            top_prediction = predictions[0] if predictions else None
            
            return {
                'predictions': predictions,
                'top_class': top_prediction['label'] if top_prediction else None,
                'confidence': top_prediction['score'] if top_prediction else 0.0,
                'anomaly_detected': top_prediction['score'] < self.confidence_threshold if top_prediction else False
            }
            
        except Exception as e:
            print(f"Classification error: {e}")
            return {'predictions': [], 'top_class': None, 'confidence': 0.0}
    
    def draw_results(self, frame, results):
        """Draw classification results"""
        output_frame = frame.copy()
        
        if results['top_class']:
            # Show top prediction
            text = f"Class: {results['top_class']} ({results['confidence']:.2f})"
            cv2.putText(output_frame, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show anomaly status if applicable
            if 'anomaly_detected' in results:
                status = "ANOMALY DETECTED" if results['anomaly_detected'] else "NORMAL"
                color = (0, 0, 255) if results['anomaly_detected'] else (0, 255, 0)
                cv2.putText(output_frame, status, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return output_frame


class ImageSegmentationProcessor(BaseVisionProcessor):
    """Image segmentation for detailed pixel-level analysis"""
    
    def get_task_type(self):
        return "image-segmentation"
    
    def process_frame(self, frame):
        """Process frame for segmentation"""
        if not self.is_available():
            return {'segments': [], 'masks': None}
        
        try:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            segments = self.pipeline(pil_image)
            
            # Filter by confidence
            filtered_segments = [
                seg for seg in segments 
                if seg['score'] >= self.confidence_threshold
            ]
            
            return {
                'segments': filtered_segments,
                'raw_segments': segments,
                'count': len(filtered_segments)
            }
            
        except Exception as e:
            print(f"Segmentation error: {e}")
            return {'segments': [], 'masks': None}
    
    def draw_results(self, frame, results):
        """Draw segmentation masks"""
        output_frame = frame.copy()
        
        for i, segment in enumerate(results['segments']):
            # Convert PIL mask to numpy
            mask = np.array(segment['mask'])
            
            # Create colored mask
            color = np.random.randint(0, 255, 3).tolist()
            colored_mask = np.zeros_like(output_frame)
            colored_mask[mask] = color
            
            # Blend with original image
            output_frame = cv2.addWeighted(output_frame, 0.7, colored_mask, 0.3, 0)
            
            # Add label
            label = f"{segment['label']}: {segment['score']:.2f}"
            cv2.putText(output_frame, label, (10, 30 + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return output_frame


class DepthEstimationProcessor(BaseVisionProcessor):
    """Depth estimation for 3D understanding"""
    
    def get_task_type(self):
        return "depth-estimation"
    
    def process_frame(self, frame):
        """Process frame for depth estimation"""
        if not self.is_available():
            return {'depth_map': None, 'min_depth': 0, 'max_depth': 0}
        
        try:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = self.pipeline(pil_image)
            
            depth_map = np.array(result['depth'])
            
            return {
                'depth_map': depth_map,
                'min_depth': depth_map.min(),
                'max_depth': depth_map.max()
            }
            
        except Exception as e:
            print(f"Depth estimation error: {e}")
            return {'depth_map': None, 'min_depth': 0, 'max_depth': 0}
    
    def draw_results(self, frame, results):
        """Draw depth visualization"""
        if results['depth_map'] is None:
            return frame
        
        # Normalize depth map to 0-255
        depth_normalized = cv2.normalize(results['depth_map'], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Resize to match frame size
        depth_resized = cv2.resize(depth_normalized, (frame.shape[1], frame.shape[0]))
        
        # Apply color map
        depth_colored = cv2.applyColorMap(depth_resized, cv2.COLORMAP_PLASMA)
        
        # Combine with original frame
        output_frame = cv2.addWeighted(frame, 0.6, depth_colored, 0.4, 0)
        
        # Add depth info
        depth_info = f"Depth: {results['min_depth']:.2f} - {results['max_depth']:.2f}"
        cv2.putText(output_frame, depth_info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return output_frame


class VisionModelFactory:
    """Factory class to create appropriate vision processors"""
    
    PROCESSORS = {
        'object-detection': ObjectDetectionProcessor,
        'image-classification': ImageClassificationProcessor,
        'image-segmentation': ImageSegmentationProcessor,
        'depth-estimation': DepthEstimationProcessor,
    }
    
    @classmethod
    def create_processor(cls, task_type, model_name, confidence_threshold=0.3, device=None):
        """Create a vision processor for the specified task"""
        if task_type not in cls.PROCESSORS:
            raise ValueError(f"Unsupported task type: {task_type}. Supported: {list(cls.PROCESSORS.keys())}")
        
        processor_class = cls.PROCESSORS[task_type]
        return processor_class(model_name, confidence_threshold, device)
    
    @classmethod
    def list_supported_tasks(cls):
        """List all supported task types"""
        return list(cls.PROCESSORS.keys())


# Updated main model interface
class FlexibleVisionDetector:
    """
    Flexible vision detector that can work with any Hugging Face vision model
    """
    
    def __init__(self, task_type, model_name, confidence_threshold=0.3):
        self.task_type = task_type
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        
        # Create appropriate processor
        self.processor = VisionModelFactory.create_processor(
            task_type, model_name, confidence_threshold
        )
        
        print(f"Initialized {task_type} detector with model: {model_name}")
    
    def process_frame(self, frame):
        """Process a frame using the configured model"""
        return self.processor.process_frame(frame)
    
    def draw_detections(self, frame, results):
        """Draw detection results on frame"""
        return self.processor.draw_results(frame, results)
    
    def is_available(self):
        """Check if the model is available"""
        return self.processor.is_available()


# Example usage configurations
EXAMPLE_MODELS = {
    'object-detection': [
        "facebook/detr-resnet-50",
        "hustvl/yolos-tiny",
        "microsoft/table-transformer-detection"
    ],
    'image-classification': [
        "microsoft/resnet-50",
        "google/vit-base-patch16-224",
        "microsoft/DiNAT-Large-ImageNet-1K-224"
    ],
    'image-segmentation': [
        "facebook/detr-resnet-50-panoptic",
        "nvidia/segformer-b0-finetuned-ade-512-512",
        "facebook/mask2former-swin-base-coco-panoptic"
    ],
    'depth-estimation': [
        "Intel/dpt-large",
        "facebook/dpt-dinov2-small-nyu",
        "LiheYoung/depth-anything-small-hf"
    ]
}