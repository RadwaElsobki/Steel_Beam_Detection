import cv2
import torch
from groundingdino.util.inference import Model
from groundingdino.datasets.transforms import resize

# Load the GroundingDINO model
config_path = '/path/to/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
weights_path = '/path/to/GroundingDINO/weights/groundingdino_swint_ogc.pth'
model = Model(config_path, weights_path)
model.eval()

# Define the transformation
transforms = resize()

# Path to the input image
image_path = '/home/radwa/project/data/steel_beam_09.jpg'
output_path = '/home/radwa/project/data/output_detected_beams_annotated.jpg'

# Load the image
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Unable to load image at {image_path}")
    exit()

# Preprocess the image
input_image = transforms(image)
input_image = input_image.unsqueeze(0)  # Add batch dimension

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_image = input_image.to(device)
model.to(device)

# Run inference
with torch.no_grad():
    outputs = model(input_image)

# Extract detections and draw them on the image
boxes = outputs[0]['boxes'].cpu().numpy()
scores = outputs[0]['scores'].cpu().numpy()
labels = outputs[0]['labels'].cpu().numpy()

for box, score, label in zip(boxes, scores, labels):
    if score > 0.5:  # Confidence threshold
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, f'Label: {label}, Score: {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Save the annotated image to a file
cv2.imwrite(output_path, image)
print(f"Processed and annotated image saved to {output_path}")

# Display the result
cv2.imshow('Detected Steel Beams', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
