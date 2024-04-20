import numpy as np
import cv2
from queue import Queue
import matplotlib.pyplot as plt


# Implementing region growing algorithm
def segment_object(image, seed_points, threshold_value):
    # Get the dimensions of the input image.
    height, width = image.shape

    # Create a queue
    queue = Queue()

    # Create a mask to mark visited pixels
    visited_pixels = np.zeros_like(image, dtype=np.uint8)

    # Store the segmented region.
    segmented_image = np.zeros_like(image, dtype=np.uint8)

    # Add seed points to the queue and mark visited seeds.
    for seed_point in seed_points:
        queue.put(seed_point)
        visited_pixels[seed_point] = 1

    # Define neighbors.
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Perform region growing.
    while not queue.empty():
        current_pixel = queue.get()
        x, y = current_pixel

        segmented_image[x, y] = 255

        # Find neighbors.
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy

            # Check if neighbor is within image bounds.
            if 0 <= nx < height and 0 <= ny < width:
                # Check if neighbor is not visited.
                if not visited_pixels[nx, ny]:
                    # Check pixel intensity difference against threshold.
                    if abs(int(image[nx, ny]) - int(image[x, y])) < threshold_value:
                        queue.put((nx, ny))
                        visited_pixels[nx, ny] = 1

    return segmented_image



if __name__ == "__main__":
    # Read the image from the path.
    image = cv2.imread("Test_images/Question_2/Image.jpeg", cv2.IMREAD_GRAYSCALE)

    # Define seed points
    seed_points = [(500, 300), (600, 250)]

    # Define the intensity threshold
    threshold_value = 3

    # Perform region growing segmentation.
    segmented_image = segment_object(image, seed_points, threshold_value)

    cv2.imwrite("Result_images/Question_2/Segmented_image.jpeg", segmented_image)

    # Display output.
    plt.figure("Segmented Image")
    plt.imshow(segmented_image, cmap='gray')
    plt.axis('off')
    plt.show()
