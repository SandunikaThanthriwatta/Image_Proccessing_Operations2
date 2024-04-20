import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create an image with 2 objects and 3 pixel values
def image_generate(path):
    # Generate a NumPy array with a black background
    image = np.zeros((200,200), dtype=np.uint8)

    # Draw two circles on black background.
    cv2.circle(image, (130,50), 30, 200, -1)
    cv2.circle(image, (50, 140), 20, 240, -1)

    # Save the image
    cv2.imwrite(path, image)

# Add Gaussian noise to the image.
def gaussian_noise(image, mean, std):
    # Generate Gaussian noise defining mean and standard deviation.
    gaussian_noise = np.random.normal(mean, std, image.shape)

    # Add the generated noise to the image
    image_with_noise = np.clip(image + gaussian_noise, 0, 255).astype(np.uint8)
    return image_with_noise

# Implementation of Otsu's algorithm for image thresholding.
def otsu_algorithm(image):
    # Calculate the histogram of the input
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    # Calculate the number of pixels
    total_pixels = image.shape[0] * image.shape[1]

    # Initialize variables .
    sumT = 0
    sumB = 0
    wB = 0
    wF = 0
    varMax = 0
    threshold = 0

    # Iterate through all possible threshold values.
    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total_pixels - wB
        if wF == 0:
            break

        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sumT - sumB) / wF


        varBetween = wB * wF * (mB - mF) * (mB - mF)

        # Update the threshold
        if varBetween > varMax:
            varMax = varBetween
            threshold = t

        sumT += hist[t]

    # Apply the threshold to the image
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    return binary

if __name__ == "__main__":
    # Define the path to the input image.
    image_path = 'Test_images/Question_1/Q1_test_image.jpg'

   #call functions
    image_generate(image_path)
    original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    noisy_img = gaussian_noise(original_img, mean=0, std=60)
    resulted_img = otsu_algorithm(noisy_img)

    # Save the noisy image to disk.
    cv2.imwrite("Result_images/Question_1/Noisy_image.jpg", noisy_img)

    # Save the Otsu segmented image to disk.
    cv2.imwrite("Result_images/Question_1/Otsu_image.jpg", resulted_img)

    # Display the results.
    plt.figure("Otsu Thresholded Image", figsize=(10, 4))
    plt.subplot(131)
    plt.imshow(original_img, cmap='gray')
    plt.title('Original Image')
    plt.subplot(132)
    plt.imshow(noisy_img, cmap='gray')
    plt.title('Image After Adding Gaussian Noise')
    plt.subplot(133)
    plt.imshow(resulted_img, cmap='gray')
    plt.title('Otsu Thresholded Image')
    plt.tight_layout()
    plt.show()
