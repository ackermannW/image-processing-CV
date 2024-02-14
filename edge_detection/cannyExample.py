import cv2 as cv
import numpy as np
import os 
import sys
import matplotlib.pyplot as plt

script_path = os.path.abspath(sys.argv[0])
path = os.path.abspath(os.path.join(os.path.dirname(script_path), '..', 'images', 'm1a2_abrams_l5.jpg'))
image = cv.imread(path)

# Read the image

# Convert the image to grayscale
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv.GaussianBlur(gray, (3, 3), 0)

# Perform Canny edge detection
edges = cv.Canny(blurred, 50, 170)

# Create a mask of the edges
mask = cv.bitwise_and(image, image, mask=edges)
# Combine the mask with the original image to create a sharpened result
sharpened = cv.addWeighted(image, 1.5, mask, -0.5, 0)

# Display the original and sharpened images

fig, axs = plt.subplots(1, 2)

axs[0].imshow(image)
axs[0].set_title('Original image')

axs[1].imshow(sharpened)
axs[1].set_title('Sharpened image')

fig.show()

plt.tight_layout()
plt.show()

print("Press 0 to close.")
cv.waitKey(0)
cv.destroyAllWindows()
