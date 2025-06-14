cat > README.md << 'EOF'
# Steel Beam Detection Project

## Overview

This project explores methods for detecting steel beams on construction sites to assist in work follow-up, as-built recognition, and deconstruction processes. The core approach includes experimenting with state-of-the-art object detection and segmentation models such as **YOLOv8**, **GroundingDINO**, and **Segment Anything Model (SAM)**, as well as traditional computer vision techniques like **Canny edge detection** and **corner detection** to estimate beam locations in 3D space using a ZED depth camera.

## Project Structure

- `YOLOv8_trials/`: Contains trials and experiments with the YOLOv8 model for beam detection.
- `assets/`: Includes supporting assets, datasets, and pre/post-processing scripts.
- `Edge_detection/`: Contains results and code related to Canny edge and corner detection methods.
- `output/`: Output images and results from different model trials.
- `GroundingDINO_SAM_trial/`: Trials involving GroundingDINO, SAM, and other models.

## Important Notes

- **SAM (Segment Anything Model)** and **GroundingDINO** models are referenced and used in this project for segmentation and grounding tasks.  
- These models themselves are **not included** in this repository due to their large size.  
- To use these models, please refer to their official repositories:  
  - [SAM GitHub Repo](https://github.com/facebookresearch/segment-anything)  
  - [GroundingDINO GitHub Repo](https://github.com/IDEA-Research/GroundingDINO)  

You can download the model weights and code directly from their original sources.



