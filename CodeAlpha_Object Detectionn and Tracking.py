 #!/usr/bin/env python3
"""
Object Detection and Tracking with YOLO + Deep SORT
This script automatically installs required packages and runs real-time tracking
"""
import subprocess
import sys
import os

def install_packages():
    """Automatically install required packages"""
    required_packages = [
        'opencv-python',
        'ultralytics',
        'deep-sort-realtime',
        'numpy',
        'scipy',
        'filterpy'
    ]
    
    print("=" * 60)
    print("Checking and installing required packages...")
    print("=" * 60)
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                __import__('cv2')
            elif package == 'ultralytics':
                __import__('ultralytics')
            elif package == 'deep-sort-realtime':
                __import__('deep_sort_realtime')
            else:
                __import__(package)
            print(f"✓ {package} is already installed")
        except ImportError:
            print(f"📦 Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")
    
    print("=" * 60)
    print("All packages ready!\n")

# Install packages before importing
install_packages()

# Now import all libraries
import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# COCO class names (80 classes that YOLO detects)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "fan"
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "door"
   "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "windows",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "table", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "plants", "tree",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "mobile phone", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def main():
    print("=" * 60)
    print("OBJECT DETECTION AND TRACKING SYSTEM")
    print("=" * 60)
    print("\nInitializing...")
    
    # Load YOLO model (downloads automatically on first run)
    print("Loading YOLO model (first time may download weights)...")
    model = YOLO("yolov8n.pt")  # Using nano version for speed
    print("✓ YOLO model loaded")
    
    # Initialize Deep SORT tracker
    print("Initializing Deep SORT tracker...")
    tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)
    print("✓ Tracker ready")
    
    # Open webcam
    print("\nOpening webcam (camera 0)...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam!")
        print("   Try changing camera index to 1 or -1 in the code")
        return
    
    print("✓ Webcam opened successfully")
    print("\n" + "=" * 60)
    print("TRACKING ACTIVE")
    print("- Press 'q' to quit")
    print("- Press 's' to save current frame")
    print("- Press 'r' to reset tracker")
    print("=" * 60 + "\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame_count += 1
        
        # Run detection every frame (for smooth tracking)
        results = model(frame, verbose=False)
        
        # Prepare detections for tracker
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Only track objects with confidence > 0.5
                    if conf > 0.5:
                        bbox = [x1, y1, x2 - x1, y2 - y1]  # [left, top, width, height]
                        detections.append((bbox, conf, cls))
        
        # Update tracker
        tracked_objects = tracker.update_tracks(detections, frame=frame)
        
        # Draw results
        for track in tracked_objects:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            cls = track.get_det_class()
            
            # Get class name
            class_name = COCO_CLASSES[cls] if cls < len(COCO_CLASSES) else f"Class_{cls}"
            
            # Color based on track ID (for visual variety)
            color = tuple(hash(track_id) % 255 for _ in range(3))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{class_name} #{track_id}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display info
        cv2.putText(frame, f"Objects: {len(tracked_objects)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow("Object Detection and Tracking", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            filename = f"screenshot_{frame_count}.png"
            cv2.imwrite(filename, frame)
            print(f"📸 Screenshot saved as {filename}")
        elif key == ord('r'):
            tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)
            print("🔄 Tracker reset")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Program ended successfully")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Program interrupted by user")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have a working webcam")
        print("2. Try changing 'cv2.VideoCapture(0)' to 'cv2.VideoCapture(1)'")
        print("3. Install Python 3.11 or 3.12 if you're using 3.13")
        input("\nPress Enter to exit...")
        