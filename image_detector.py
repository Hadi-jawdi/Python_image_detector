"""
Image Detection Application using YOLOv8
Detects objects in images and displays/saves results
"""

import cv2  
import os
from pathlib import Path
from ultralytics import YOLO  # type: ignore
import argparse


class ImageDetector:
    """Image detection class using YOLOv8 model"""
    
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize the detector with a YOLO model
        
        Args:
            model_path: Path to YOLO model file (default: yolov8n.pt - nano model)
        """
        print(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        print("Model loaded successfully!")
    
    def detect_image(self, image_path, conf_threshold=0.25, save=True, show=True):
        """
        Detect objects in a single image
        
        Args:
            image_path: Path to the input image
            conf_threshold: Confidence threshold (0-1)
            save: Whether to save the annotated image
            show: Whether to display the image
            
        Returns:
            Detection results
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"\nDetecting objects in: {image_path}")
        
        # Run detection
        results = self.model(image_path, conf=conf_threshold)
        
        # Process results
        annotated_image = results[0].plot()
        
        # Display detected objects info
        detections = results[0].boxes
        if len(detections) > 0:
            print(f"\nFound {len(detections)} object(s):")
            for i, box in enumerate(detections):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names[class_id]
                print(f"  {i+1}. {class_name}: {confidence:.2%} confidence")
        else:
            print("No objects detected.")
        
        # Save annotated image
        if save:
            output_path = self._get_output_path(image_path)
            cv2.imwrite(output_path, annotated_image)
            print(f"\nAnnotated image saved to: {output_path}")
        
        # Display image
        if show:
            cv2.imshow('Detection Results', annotated_image)
            print("\nPress any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return results
    
    def detect_batch(self, image_dir, conf_threshold=0.25, save=True):
        """
        Detect objects in multiple images in a directory
        
        Args:
            image_dir: Directory containing images
            conf_threshold: Confidence threshold (0-1)
            save: Whether to save annotated images
            
        Returns:
            List of detection results
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [
            f for f in os.listdir(image_dir)
            if Path(f).suffix.lower() in image_extensions
        ]
        
        if not image_files:
            print(f"No image files found in {image_dir}")
            return []
        
        print(f"\nProcessing {len(image_files)} image(s)...")
        results = []
        
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            try:
                result = self.detect_image(
                    image_path, 
                    conf_threshold=conf_threshold, 
                    save=save, 
                    show=False
                )
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
        
        return results
    
    def detect_webcam(self, conf_threshold=0.25):
        """
        Real-time detection from webcam
        
        Args:
            conf_threshold: Confidence threshold (0-1)
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("\nStarting webcam detection...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = self.model(frame, conf=conf_threshold)
            annotated_frame = results[0].plot()
            
            # Display frame
            cv2.imshow('Webcam Detection', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam detection stopped.")
    
    def _get_output_path(self, image_path):
        """Generate output path for annotated image"""
        path = Path(image_path)
        output_path = path.parent / f"{path.stem}_detected{path.suffix}"
        return str(output_path)


def main():
    """Main function to run the image detector"""
    parser = argparse.ArgumentParser(description='Image Detection using YOLOv8')
    parser.add_argument('--image', '-i', type=str, help='Path to input image')
    parser.add_argument('--directory', '-d', type=str, help='Directory containing images')
    parser.add_argument('--webcam', '-w', action='store_true', help='Use webcam for detection')
    parser.add_argument('--model', '-m', type=str, default='yolov8n.pt', 
                       help='Path to YOLO model (default: yolov8n.pt)')
    parser.add_argument('--confidence', '-c', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save annotated images')
    parser.add_argument('--no-show', action='store_true', help='Don\'t display images')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = ImageDetector(model_path=args.model)
    
    # Run detection based on input
    if args.webcam:
        detector.detect_webcam(conf_threshold=args.confidence)
    elif args.directory:
        detector.detect_batch(
            args.directory, 
            conf_threshold=args.confidence,
            save=not args.no_save
        )
    elif args.image:
        detector.detect_image(
            args.image,
            conf_threshold=args.confidence,
            save=not args.no_save,
            show=not args.no_show
        )
    else:
        # Interactive mode - ask user for input
        print("\n=== Image Detection Application ===")
        print("Select mode:")
        print("1. Detect objects in a single image")
        print("2. Detect objects in multiple images (directory)")
        print("3. Real-time webcam detection")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            image_path = input("Enter image path: ").strip()
            detector.detect_image(image_path, conf_threshold=args.confidence)
        elif choice == '2':
            dir_path = input("Enter directory path: ").strip()
            detector.detect_batch(dir_path, conf_threshold=args.confidence)
        elif choice == '3':
            detector.detect_webcam(conf_threshold=args.confidence)
        else:
            print("Invalid choice!")


if __name__ == '__main__':
    main()
