import cv2
import matplotlib.pyplot as plt
import numpy as np


def count_intensity_values(image):
    # to store count of each intensity
    intensity_count = [0] * 256

    for row in image:
        for pixel in row:
            intensity = pixel
            intensity_count[intensity] += 1

    return intensity_count


image = cv2.imread('Image_Q4.tif', cv2.IMREAD_GRAYSCALE)

intensity_count = count_intensity_values(image)

plt.bar(range(256), intensity_count)
plt.xlabel('Intensity Value')
plt.ylabel('Count')
plt.title('Intensity Value Counts')
plt.show()

# part ii
original_image = cv2.imread('Image_Q4.tif', cv2.IMREAD_GRAYSCALE)

# making 7x7 and 3x3 box filters
kernel_7x7 = np.ones((7, 7), np.float32) / 49  # Normalization factor is 49
kernel_3x3 = np.ones((3, 3), np.float32) / 9   # Normalization factor is 9

# Apply the  filters to the original image
output_7x7 = cv2.filter2D(original_image, -1, kernel_7x7)
output_3x3 = cv2.filter2D(original_image, -1, kernel_3x3)

# Calculate difference between outputs of 2 filters
absolute_difference = cv2.absdiff(output_7x7, output_3x3)

# Subtract difference from original image
final_output = cv2.subtract(original_image, absolute_difference)

# Display the result
cv2.imshow('Original', original_image)
cv2.imshow('Final Output', final_output)
cv2.waitKey(0)
cv2.destroyAllWindows()
