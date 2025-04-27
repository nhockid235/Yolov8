from ultralytics import YOLO
import cv2
import os

# Create directory for results
os.makedirs('results', exist_ok=True)

def detect_image(image_path, output_name="lab2.jpg"):
    """Run YOLOv8 detection on a specific image file path"""
    print(f"Running detection on: {image_path}")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return
    
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')  # load pretrained model
    
    try:
        # Run detection
        results = model(image_path)
        
        # Process results
        for r in results:
            # Save annotated image
            im_array = r.plot()
            output_path = f'results/{output_name}'
            cv2.imwrite(output_path, im_array)
            
            # Print detected objects
            boxes = r.boxes
            print(f"\nDetected {len(boxes)} objects:")
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                print(f"  - {class_name} (confidence: {conf:.2f})")
        
        print(f"\nDetection completed. Result saved to 'results/{output_name}'")
    
    except Exception as e:
        print(f"Error during detection: {e}")

if __name__ == "__main__":
    # If you want to save the highway image from the chat, 
    # you would need to manually download it first
    
    # Assuming you've already saved the highway image manually
    highway_image_path = "highway.jpg"
    detect_image(highway_image_path) 