import os
from ultralytics import YOLO
from matplotlib import pyplot as plt
from PIL import Image as PILImage
import numpy as np

# Load the trained model
my_new_model = YOLO('/home/radwa/Documents/assets/results/test_run/weights/last.pt')

# Path to the new image for inference
new_image = '/home/radwa/Documents/assets/Data_set/Valid_images/SB_08.jpg'

# Run inference on the new image
new_results = my_new_model.predict(new_image, conf=0.3)  # Adjust confidence threshold

# Check if there are any detections
if new_results[0].masks is not None:
    # Get the mask from the results
    masks = new_results[0].masks.data  # Extract masks data

    # Create a blank black and white mask image
    mask_image = np.zeros((masks[0].shape[0], masks[0].shape[1]), dtype=np.uint8)

    # Combine all masks into one image
    for mask in masks:
        mask_image = np.maximum(mask_image, mask.cpu().numpy().astype(np.uint8))

    # Save the black and white mask image
    mask_image_pil = PILImage.fromarray(mask_image * 255)  # Convert to PIL image, scale to 255
    mask_image_path = '/home/radwa/Documents/assets/steel_beam_08_mask.png'
    mask_image_pil.save(mask_image_path)

    # Plot and display the inference results
    new_result_array = new_results[0].plot()

    # Load the original image
    original_image = PILImage.open(new_image)

    # Display the original, result, and mask images side by side
    plt.figure(figsize=(20, 15))

    # Original image
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(original_image)
    plt.axis('off')

    # Annotated image
    plt.subplot(1, 3, 2)
    plt.title('Annotated Image')
    plt.imshow(new_result_array)
    plt.axis('off')

    # Mask image
    plt.subplot(1, 3, 3)
    plt.title('Mask Image (Black and White)')
    plt.imshow(mask_image, cmap='gray')
    plt.axis('off')

    plt.show()

    print(f"Mask image saved to {mask_image_path}")
else:
    print("No detections were made in the image.")
    # Display the original image only
    original_image = PILImage.open(new_image)

    plt.figure(figsize=(7, 7))
    plt.title('Original Image - No Detections')
    plt.imshow(original_image)
    plt.axis('off')

    plt.show()
