import cv2
import numpy as np

def canny_edge_detection(image_path, output_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load the image at {image_path}. Please check the file path and format.")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    # Apply Canny edge detector
    edges = cv2.Canny(blurred, 100, 200)

    # Save the result
    cv2.imwrite(output_path, edges)
    print(f"Edge detection completed. Processed image saved at {output_path}")

# Example usage
image_path = "/home/radwa/Documents/assets/steel_beam_05.jpg"
output_path = "/home/radwa/Documents/output/detect_beam05.jpg"
canny_edge_detection(image_path, output_path)
