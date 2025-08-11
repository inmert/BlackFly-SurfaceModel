import cv2
import numpy as np
from transformers import pipeline, AutoImageProcessor, AutoModelForImageSegmentation
import torch
from PIL import Image

class CrackDetector:
    """
    Crack detection using Hugging Face models
    Uses segmentation models for pixel-level anomaly detection
    """
    
    def __init__(self, model_name="microsoft/DiNAT-Large-ImageNet-1K-224", confidence_threshold=0.3):
        self.confidence_threshold = confidence_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize the model pipeline
        try:
            # For crack detection, we'll use an image classification model
            # and treat low-confidence predictions as potential anomalies
            self.classifier = pipeline(
                "image-classification",
                model=model_name,
                device=0 if self.device == "cuda" else -1
            )
            print(f"Loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            # Fallback to a simpler approach using traditional CV
            self.classifier = None
            print("Falling back to traditional computer vision methods")
    
    def detect_anomalies_cv(self, frame):
        """
        Traditional computer vision approach for crack detection
        Good fallback when ML models aren't available
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection using Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Morphological operations to connect broken edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on area and aspect ratio (typical for cracks)
        crack_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Minimum area threshold
                # Calculate bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                
                # Cracks typically have high aspect ratio
                if aspect_ratio > 3:
                    crack_contours.append(contour)
        
        return crack_contours, edges
    
    def detect_anomalies_ml(self, frame):
        """
        ML-based anomaly detection using Hugging Face model
        """
        # Convert frame to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        try:
            # Get predictions
            predictions = self.classifier(pil_image)
            
            # Look for low-confidence predictions or specific classes that might indicate damage
            anomaly_score = 1.0 - max([p['score'] for p in predictions])
            
            # Simple threshold-based detection
            has_anomaly = anomaly_score > self.confidence_threshold
            
            return has_anomaly, anomaly_score, predictions
            
        except Exception as e:
            print(f"ML detection error: {e}")
            return False, 0.0, []
    
    def process_frame(self, frame):
        """
        Main processing function that combines both approaches
        """
        results = {
            'anomalies_detected': False,
            'confidence': 0.0,
            'contours': [],
            'processed_frame': frame.copy()
        }
        
        if self.classifier is not None:
            # Use ML approach
            has_anomaly, confidence, predictions = self.detect_anomalies_ml(frame)
            results['anomalies_detected'] = has_anomaly
            results['confidence'] = confidence
            results['ml_predictions'] = predictions
            
            # If anomaly detected, also run CV method for visualization
            if has_anomaly:
                contours, edges = self.detect_anomalies_cv(frame)
                results['contours'] = contours
        else:
            # Use traditional CV approach
            contours, edges = self.detect_anomalies_cv(frame)
            results['contours'] = contours
            results['anomalies_detected'] = len(contours) > 0
            results['confidence'] = len(contours) / 10.0  # Rough confidence based on number of contours
        
        return results
    
    def draw_detections(self, frame, results):
        """
        Draw detection results on the frame
        """
        output_frame = frame.copy()
        
        if results['anomalies_detected']:
            # Draw contours if available
            if results['contours']:
                cv2.drawContours(output_frame, results['contours'], -1, (0, 0, 255), 2)
                
                # Draw bounding boxes around detected areas
                for contour in results['contours']:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(output_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Add status text
            status_text = f"ANOMALY DETECTED (Conf: {results['confidence']:.2f})"
            cv2.putText(output_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(output_frame, "NO ANOMALIES", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return output_frame