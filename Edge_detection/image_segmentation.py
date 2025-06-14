import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry

# Load the image
image_path = '/home/radwa/Documents/assets/steel_beam_03.jpg'
image = cv2.imread(image_path)

# Initialize the SAM model
sam = sam_model_registry["vit_h"](checkpoint="/home/radwa/Documents/sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)

# Set the image in the predictor
predictor.set_image(image)

# Get segmentation mask
masks = predictor.predict()

# Assuming masks[0] contains the actual mask
if isinstance(masks, tuple):
    masks = masks[0]

# Debugging: Check the shape and type of the mask
print("Mask shape:", masks.shape)
print("Mask dtype:", masks.dtype)
print("Mask unique values:", np.unique(masks))

# Ensure the mask is 2D if it is 3D
if masks.ndim == 3:
    masks = masks[:, :, 0]

# Convert the mask to a format suitable for edge detection
masks = (masks * 255).astype(np.uint8)

# Debugging: Check the shape and type of the mask after conversion
print("Mask shape after conversion:", masks.shape)
print("Mask dtype after conversion:", masks.dtype)
print("Mask unique values after conversion:", np.unique(masks))

# Use Sobel filter for initial edge detection
sobelx = cv2.Sobel(masks, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(masks, cv2.CV_64F, 0, 1, ksize=5)
sobel_edges = cv2.magnitude(sobelx, sobely)
sobel_edges = cv2.convertScaleAbs(sobel_edges)

# Refine edges using Canny filter
canny_edges = cv2.Canny(sobel_edges, threshold1=50, threshold2=150)

# Find contours
contours, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    # Get the moments to calculate the center of the contour
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    # Get the orientation using the fit ellipse method
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        angle = ellipse[2]

        # Draw the center and orientation on the image
        cv2.circle(image, (cX, cY), 5, (255, 0, 0), -1)
        cv2.ellipse(image, ellipse, (0, 255, 0), 2)

# Display the results
cv2.imshow("Segmented Image", image)
cv2.imshow("Sobel Edges", sobel_edges)
cv2.imshow("Canny Edges", canny_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
