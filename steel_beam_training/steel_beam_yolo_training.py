import os
from ultralytics import YOLO
from matplotlib import pyplot as plt
from PIL import Image
import yaml

# Load the model configuration and pretrained weights
#Instance
model = YOLO('yolov8n-seg.yaml')  # build a new model from YAML
model = YOLO('yolov8n-seg.pt')  # Load the pretrained model

# Load number of classes from the YAML file
yaml_file = "/home/radwa/Documents/assets/yolo_data.yolov8/data.yaml"
with open(yaml_file, 'r') as stream:
    data_config = yaml.safe_load(stream)
    num_classes = data_config['nc']

# Define project and training settings
project = "/home/radwa/Documents/assets/results"
name = "test_run"  # Name for this specific test run

# Ensure the base results directory exists
if not os.path.exists(project):
    os.makedirs(project)

# Train the model
results = model.train(data=yaml_file,
                      project=project,
                      name=name,
                      epochs=150,  # Reduced for quick testing
                      patience=0,  # Disable early stopping
                      batch=2,  # Reduced batch size for small dataset
                      imgsz=800)

# Save training results
results.save()

# Optional: Plot training results
results.plot()
plt.show()
