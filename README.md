# Real-time Surface Detection System

This project is a modular and extensible application for performing real-time computer vision detection on video streams. It supports various camera sources and can be configured to use different detection models, from traditional computer vision algorithms to state-of-the-art deep learning models like YOLOv8 and DETR.

## Features

* **Real-time Processing:** Designed for low-latency video analysis.
* **Multiple Camera Support:** Seamlessly switch between Blackfly industrial cameras and standard webcams.
* **Configurable Models:** Easily select and configure different detection models for various tasks:
    * Traditional Computer Vision (e.g., crack detection).
    * Object Detection (e.g., aircraft detection with YOLOv8, general objects with DETR).
    * Instance Segmentation (e.g., with Mask2Former).
* **GPU Acceleration:** Automatically leverages CUDA (NVIDIA), MPS (Apple Silicon), or falls back to CPU for efficient processing.
* **Modular Architecture:** The system is built with a clear separation of concerns, making it easy to add new camera sources or detection models.

## Getting Started

### Prerequisites

* Python 3.8 or higher
* `pip` for package management
* A compatible camera (Webcam or Blackfly)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/BlackFly-SurfaceModel.git](https://github.com/your-username/BlackFly-SurfaceModel.git)
    cd BlackFly-SurfaceModel
    ```

2.  **Create and activate a virtual environment:**
    It is highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    python -m venv .venv
    # On macOS/Linux:
    source .venv/bin/activate
    # On Windows:
    .\.venv\Scripts\activate
    ```

3.  **Install dependencies:**
    The project uses several libraries. The following command installs the core dependencies, including those for Blackfly cameras, OpenCV, and the deep learning frameworks for all models.

    ```bash
    pip install -r requirements.txt
    ```
    *Note: The `requirements.txt` file is not included in the provided code, but should contain the following libraries:*
    `opencv-python`
    `numpy`
    `torch`
    `ultralytics`
    `transformers`
    `blackfly_sdk` (or similar for Blackfly cameras)

## Configuration

All system, camera, and model settings are managed in the `config.py` file.

### Selecting a Detector

You can choose which detection model to use by modifying the `--detector` argument when running the application. The available detectors are defined in `config.py`:
* `crack_detection`
* `object_detection` (using DETR)
* `yolov8_aircraft_detection` (using YOLOv8)
* `segmentation`

### Example: Using YOLOv8 for Aircraft Detection

To use the faster YOLOv8 model for aircraft detection, the `yolov8_aircraft_detection` configuration in `config.py` is crucial:
```python
'yolov8_aircraft_detection': {
    'type': 'object_detection',
    'model_name': 'yolov8n.pt',  # YOLOv8 nano model for high speed
    'confidence_threshold': 0.4,
    'target_classes': ['airplane'], # Filter for a specific class
    'framework': 'ultralytics', # Specify the framework for loading
},