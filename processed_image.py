import cv2
import numpy as np
import base64
import os
from PIL import Image
import io

# Save the example image locally
def save_sample_image():
    # URL of a highway scene - let's use this since we can't directly access the uploaded image
    url = "https://ultralytics.com/images/zidane.jpg"
    
    # Use OpenCV to download and save the image
    import urllib.request
    
    # Create directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    # Download the image
    try:
        urllib.request.urlretrieve(url, 'images/highway_image.jpg')
        print(f"Sample image downloaded and saved to 'images/highway_image.jpg'")
        return 'images/highway_image.jpg'
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

if __name__ == "__main__":
    filepath = save_sample_image()
    if filepath:
        print(f"Image saved successfully to {filepath}")
    else:
        print("Failed to save image.") 