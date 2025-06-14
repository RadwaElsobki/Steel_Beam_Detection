import cv2
import numpy as np

# Load an example depth image
image = cv2.imread('/home/radwa/project/data/steel_beam_09.jpg', cv2.IMREAD_GRAYSCALE)

# Apply GaussianBlur to reduce noise
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Use Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours and calculate bounding boxes
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Calculate position and orientation
    position = (x + w // 2, y + h // 2)
    orientation = np.arctan2(h, w) * 180 / np.pi
    print(f"Position: {position}, Orientation: {orientation}")

# Display the result
cv2.imshow('Detected Steel Beams', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
