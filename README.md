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

## YOLOv8 Test Run Output

Here is an example of YOLOv8 detecting steel beams on the construction site:

![YOLOv8 Steel Beam Test Run Output](/output/YOLO8_results/test_run/train_batch0.jpg)

*Note: Data segmentation for training was done using [Labelbox](https://labelbox.com/).*

You can find the official YOLOv8 repository here:  
[YOLOv8 on Ultralytics GitHub](https://github.com/ultralytics/ultralytics)


## Dataset

**Data Annotation and Dataset Preparation**  
I used **[Roboflow](https://roboflow.com/)** for image classification and segmentation of steel beams. Roboflow was especially useful for downloading and exporting the dataset in **YOLOv8 JSON format**, which streamlined the training process.  
**[Labelbox](https://labelbox.com/)** was also used for trial data segmentation but Roboflow became my primary tool for preparing the dataset.

The steel beams dataset used in this project is available on **[Roboflow](https://app.roboflow.com/radwa/steel-beams-7f1rm/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)** for download and exploration. Roboflow is especially useful for exporting the dataset in the YOLOv8 JSON format compatible with model training.

> **Note:** This project would greatly benefit from more real on-site construction data to be collected and used for training. Additional real-world data will improve the model’s accuracy and reliability for practical applications.

## Learning Resources for Customizing YOLO8 and SAM

I learned how to customize YOLOv8 from these tutorials:  
- [Tutorial 1](https://www.youtube.com/watch?v=ytlhMAF6ok0&t=771s)  
- [Tutorial 2](https://www.youtube.com/watch?v=JQ_RRcHLKFc)  

I customized SAM following this tutorial:  
- [SAM Customization Tutorial](https://www.youtube.com/watch?v=83tnWs_YBRQ&t=1285s)  

**Data Annotation and Dataset Preparation**  
I used **[Roboflow](https://roboflow.com/)** for image classification and segmentation of steel beams. Roboflow was especially useful for downloading and exporting the dataset in **YOLOv8 JSON format**, which streamlined the training process.  
**[Labelbox](https://labelbox.com/)** was also used for trial data segmentation but Roboflow became my primary tool for preparing the dataset.


The official YOLOv8 repository can be found here:  
[YOLOv8 on Ultralytics GitHub](https://github.com/ultralytics/ultralytics)

## Important Notes

- **SAM (Segment Anything Model)** and **GroundingDINO** models are referenced and used in this project for segmentation and grounding tasks.  
- These models themselves are **not included** in this repository due to their large size.  
- To use these models, please refer to their official repositories:  
  - [SAM GitHub Repo](https://github.com/facebookresearch/segment-anything)  
  - [GroundingDINO GitHub Repo](https://github.com/IDEA-Research/GroundingDINO)  

You can download the model weights and code directly from their original sources.

---

### Workflow for Shape Localization and Robot Positioning

1. **Input Image**  
   ↓  
2. **Apply Canny Edge Detection & Other Edge Detection Methods**  
   ↓  
3. **Extract Shape Centroid Location in Image Coordinates**  
   ↓  
4. **Convert Image Coordinates to Camera Coordinates**  
   ↓  
5. **Apply Real-time Inverse Kinematics (IK) Transformation Matrix**  
   ↓  
6. **Obtain Machine/Robot Coordinates for Operation or Deconstruction**





