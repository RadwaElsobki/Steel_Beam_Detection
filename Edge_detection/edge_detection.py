import cv2
import os
import numpy as np

# Define input and output directories
inputDir = '/home/radwa/Documents/assets'
outputDir = '/home/radwa/Documents/output'
os.makedirs(outputDir, exist_ok=True)

# List all files in the input directory
files = os.listdir(inputDir)

# Loop through each file in the input directory
for file in files:
    fitem = os.path.join(inputDir, file)

    # Check if the file is an image
    if not os.path.isfile(fitem) or not fitem.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        continue

    image = cv2.imread(fitem)

    # Check if the image is loaded properly
    if image is None:
        print(f"Could not load image: {fitem}")
        continue

    # Grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Canny edge detection
    edges = cv2.Canny(gray, 150, 160, apertureSize=3)

    # Hough Line Transform
    lines = cv2.HoughLines(edges, 1.5, np.pi / 180, 200)

    # Check if any lines were detected
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            # Convert polar coordinates to Cartesian coordinates
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # Draw lines on the original image
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Define the output file path
    fout = os.path.join(outputDir, file)

    # Save the image with detected lines
    cv2.imwrite(fout, image)

    # Optionally save the grayscale image with detected edges
    gray_fout = os.path.join(outputDir, f"gray_{file}")
    cv2.imwrite(gray_fout, gray)

print(f"Processing completed. Processed images are saved in {outputDir}")
