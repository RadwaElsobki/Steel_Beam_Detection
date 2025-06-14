import numpy as np
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch

# Path to the image file
image_path = '/home/radwa/Documents/assets/steel_beam_09.jpg'

# Load the input image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image file not found at {image_path}")

# Initialize the Segment Anything Model (SAM) with the correct checkpoint
try:
    sam = sam_model_registry["vit_h"](checkpoint="/home/radwa/Documents/sam_vit_h_4b8939.pth")
except RuntimeError as e:
    print(f"Failed to load the checkpoint: {e}")
    # Attempt to download and use the correct checkpoint if necessary
    # Example: torch.hub.load('facebookresearch/detectron2', 'mask_rcnn_R_50_FPN_3x', pretrained=True)

mask_generator = SamAutomaticMaskGenerator(sam)

# Detect and segment objects in the image
masks = mask_generator.generate(image)

# Function to find the steel beam mask based on some criteria (e.g., largest area)
def find_steel_beam_mask(masks):
    # Assuming the steel beam is the largest object in the image
    largest_mask = max(masks, key=lambda x: np.sum(x["segmentation"]))
    return largest_mask

# Find the steel beam mask
steel_beam_mask = find_steel_beam_mask(masks)

# Apply the mask to the image to visualize the steel beam
segmentation = steel_beam_mask["segmentation"]
masked_image = cv2.bitwise_and(image, image, mask=segmentation.astype(np.uint8))

# Display the result
cv2.imshow("Steel Beam Detection", masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
output_path = 'home/radwa/Documents/output/steel_beam_detected.jpg'
cv2.imwrite(output_path, masked_image)
