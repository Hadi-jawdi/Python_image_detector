# Image Detection Application

A Python application for detecting objects in images using YOLOv8 (You Only Look Once) deep learning model.

## Features

- 🎯 **Object Detection**: Detect 80+ common objects (people, cars, animals, etc.)
- 📸 **Single Image Detection**: Process individual images
- 📁 **Batch Processing**: Process multiple images in a directory
- 📹 **Webcam Detection**: Real-time object detection from webcam
- 💾 **Save Results**: Automatically save annotated images with bounding boxes
- 🎨 **Visualization**: Display detection results with labels and confidence scores

## Installation

1. Install Python 3.8 or higher

2. Install required packages:
```bash
pip install -r requirements.txt
```

The first time you run the application, it will automatically download the YOLOv8 model (yolov8n.pt - about 6MB).



#### Webcam Detection
```bash
python image_detector.py --webcam
```

#### Advanced Options
```bash
# Custom confidence threshold (0.0 to 1.0)
python image_detector.py --image photo.jpg --confidence 0.5

# Use different model (yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
python image_detector.py --image photo.jpg --model yolov8s.pt

# Don't save annotated images
python image_detector.py --image photo.jpg --no-save

# Don't display images (useful for batch processing)
python image_detector.py --directory images/ --no-show
```

### Interactive Mode

Simply run without arguments to enter interactive mode:
```bash
python image_detector.py
```


## Model Options

The application supports different YOLOv8 model sizes:

- `yolov8n.pt` - Nano (fastest, least accurate) - **Default**
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (slowest, most accurate)

Larger models are more accurate but slower. The nano model is recommended for most use cases.

## Output

- Detected images are saved with `_detected` suffix (e.g., `photo.jpg` → `photo_detected.jpg`)
- Detection results show:
  - Object class name
  - Confidence score (percentage)
  - Bounding boxes with labels

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)
- WebP (.webp)

## Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLOv8
- NumPy
- Pillow


## License
This project uses the YOLOv8 model from Ultralytics, which is licensed under AGPL-3.0.


