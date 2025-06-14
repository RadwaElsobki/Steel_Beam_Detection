import cv2
import numpy as np

def main():
    # Load an example depth image
    image_path = '/home/radwa/project/data/steel_beam_09.jpg'
    output_path = '/home/radwa/project/data/output_detected_beams_annotated.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Check if the image is loaded properly
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Draw edges on the color image
    color_image[edges != 0] = [0, 0, 255]

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours and calculate bounding boxes
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(color_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Calculate position and orientation
        position = (x + w // 2, y + h // 2)
        orientation = np.arctan2(h, w) * 180 / np.pi
        print(f"Position: {position}, Orientation: {orientation}")

        # Draw the position and orientation on the image
        cv2.circle(color_image, position, 5, (0, 255, 0), -1)
        length = 50  # Length of the orientation line
        angle_rad = np.deg2rad(orientation)
        end_point = (int(position[0] + length * np.cos(angle_rad)), int(position[1] + length * np.sin(angle_rad)))
        cv2.line(color_image, position, end_point, (0, 255, 0), 2)
        cv2.putText(color_image, f"Pos: {position}, Ori: {orientation:.1f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

    try:
        # Attempt to display the image
        cv2.imshow('Detected Steel Beams with Edges', color_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error as e:
        # If there is an error (e.g., no display available), save the image to a file instead
        print(f"Error displaying image: {e}")
        cv2.imwrite(output_path, color_image)
        print(f"Processed and annotated image saved to {output_path}")

if __name__ == "__main__":
    main()

