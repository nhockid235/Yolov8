"""
YOLOv8 Object Detection with Gradio Interface

Before running this script, install the necessary dependencies:
pip install ultralytics gradio pillow

This application allows users to upload images and perform object detection
using the YOLOv8 model.
"""

import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import gradio as gr

# Ensure result directory exists
os.makedirs('results', exist_ok=True)

# Load YOLOv8 model
def load_model():
    """Load the YOLOv8 model"""
    try:
        model = YOLO('yolov8n.pt')  # Load a pretrained YOLOv8n model
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Process uploaded image
def detect_objects(image, conf_threshold=0.25):
    """
    Perform object detection on the uploaded image
    
    Args:
        image: The uploaded image
        conf_threshold: Confidence threshold for detections
    
    Returns:
        Annotated image and detection results text
    """
    if image is None:
        return None, "No image uploaded"
    
    try:
        # Convert gradio image to numpy array if it's not already
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
            
        # Convert RGB to BGR for OpenCV processing
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Save uploaded image temporarily
        temp_path = "temp_upload.jpg"
        cv2.imwrite(temp_path, image_np)
        
        # Load model
        model = load_model()
        if model is None:
            return image, "Failed to load YOLOv8 model"
        
        # Run inference
        results = model(temp_path, conf=conf_threshold)
        
        # Process results
        result_text = ""
        for r in results:
            # Get the annotated image
            annotated_img = r.plot()
            
            # Convert back to RGB for display
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            # Generate text results
            boxes = r.boxes
            result_text += f"Detected {len(boxes)} objects:\n"
            
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                result_text += f"- {class_name} (confidence: {conf:.2f})\n"
            
            # Save result with timestamp
            output_path = f"results/detection_result.jpg"
            cv2.imwrite(output_path, annotated_img)
            
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return annotated_img_rgb, result_text
        
    except Exception as e:
        error_message = f"Error processing image: {str(e)}"
        print(error_message)
        return image, error_message

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="YOLOv8 Object Detection") as app:
        gr.Markdown("# YOLOv8 Object Detection")
        gr.Markdown("Upload an image to detect objects using YOLOv8")
        
        with gr.Row():
            with gr.Column():
                # Input components
                input_image = gr.Image(type="numpy", label="Upload Image")
                conf_slider = gr.Slider(
                    minimum=0.1, 
                    maximum=1.0, 
                    value=0.25, 
                    step=0.05, 
                    label="Confidence Threshold"
                )
                detect_button = gr.Button("Detect Objects", variant="primary")
                
            with gr.Column():
                # Output components
                output_image = gr.Image(type="numpy", label="Detection Result")
                output_text = gr.Textbox(label="Detection Details")
        
        # Set up event handler
        detect_button.click(
            fn=detect_objects,
            inputs=[input_image, conf_slider],
            outputs=[output_image, output_text]
        )
        
        # Examples
        gr.Examples(
            examples=["highway.jpg", "bus.jpg"],
            inputs=input_image
        )
        
    return app

# Launch the app when script is run
if __name__ == "__main__":
    # Check if YOLOv8 model can be loaded
    model = load_model()
    if model is None:
        print("Failed to load YOLOv8 model. Please check if ultralytics is installed correctly.")
        print("Install with: pip install ultralytics")
    else:
        # Create and launch the interface
        app = create_interface()
        app.launch(share=False)  # Set share=True to create a public link 