# /home/radwa/Documents/assets/corner_detection.py

import cv2
import numpy as np

def detect_corners(image_path):
    # Load image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect corners
    corners = cv2.goodFeaturesToTrack(gray, 50, 0.01, 10)
    corners = np.intp(corners)  # Updated from np.int0 to np.intp

    # Draw corners on the image
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)

    # Display the image
    cv2.imshow('Corners', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    detect_corners("/home/radwa/Documents/assets/steel_beam_03.jpg")
