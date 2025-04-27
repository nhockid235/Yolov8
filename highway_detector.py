from ultralytics import YOLO
import cv2
import os
import urllib.request

# Create directories if they don't exist
os.makedirs('results', exist_ok=True)

def run_detection(image_path):
    """Run YOLOv8 detection on the image"""
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return False
        
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')  # load the pretrained model
    
    try:
        # Run inference
        results = model(image_path)
        
        # Process results
        for r in results:
            # Save the annotated image
            im_array = r.plot()  # plot a BGR numpy array of predictions
            output_path = 'results/lab2.jpg'
            cv2.imwrite(output_path, im_array)
            
            # Print detected objects
            boxes = r.boxes
            print(f"\nDetected {len(boxes)} objects:")
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                print(f"  - {class_name} (confidence: {conf:.2f})")
        
        print(f"\nDetection completed. Result saved to 'results/lab2.jpg'")
        return True
    
    except Exception as e:
        print(f"Error during detection: {e}")
        return False

if __name__ == "__main__":
    print("YOLOv8 Highway Image Detection")
    
    # Sử dụng tệp highway.jpg thay vì lab1.webp
    image_path = "images/lab1.jpg"
    success = run_detection(image_path)
    
    if not success:
        print("Detection failed. Please check the image file.") 