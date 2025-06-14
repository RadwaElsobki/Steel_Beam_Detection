import os
from ultralytics import YOLO
from matplotlib import pyplot as plt
from PIL import Image as PILImage

# Load the trained model
my_new_model = YOLO('/home/radwa/Documents/assets/results/test_run2/weights/last.pt')

# Path to the new image for inference
new_image = '/home/radwa/Documents/assets/steel_beam_05.jpg'

# Run inference on the new image
new_results = my_new_model.predict(new_image, conf=0.3)  # Adjust confidence threshold

# Plot and display the inference results
new_result_array = new_results[0].plot()

# Load the original image
original_image = PILImage.open(new_image)

# Display the original and result images side by side
plt.figure(figsize=(15, 15))

# Original image
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(original_image)
plt.axis('off')

# Annotated image
plt.subplot(1, 2, 2)
plt.title('Annotated Image')
plt.imshow(new_result_array)
plt.axis('off')

plt.show()

