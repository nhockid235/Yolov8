from ultralytics import YOLO
import cv2
import os

# Create a directory to save results if it doesn't exist
os.makedirs('results', exist_ok=True)

def detect_on_image(image_path=None, output_name='result_image.jpg'):
    # Load a pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')
    
    # Run inference on an image
    if image_path is None:
        # Use a sample image from Ultralytics
        image_path = 'https://ultralytics.com/images/bus.jpg'
    
    # Run inference
    results = model(image_path)
    
    # Save the results
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        output_path = f'results/{output_name}'
        cv2.imwrite(output_path, im_array)
        print(f"Image detection completed. Results saved to '{output_path}'")
        
        # Print detected objects
        boxes = r.boxes
        print(f"Detected {len(boxes)} objects:")
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls]
            print(f"  - {class_name} (confidence: {conf:.2f})")

def main():
    print("YOLOv8 Demo")
    print("Running object detection on highway image...")
    
    # Use the downloaded image
    image_path = 'images/highway_image.jpg'
    detect_on_image(image_path, 'lab2.jpg')
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main() 