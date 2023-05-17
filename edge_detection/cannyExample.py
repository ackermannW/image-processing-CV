import cv2 as cv
import numpy as np
import os 

path = os.path.join('..', os.getcwd(), 'images', 'm1a2_abrams_l5.jpg')
image = cv.imread(path)
cv.imshow('Original', image)
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
cv.imshow('Original', image)
cv.imshow('Sharpened', sharpened)
cv.waitKey(0)
cv.destroyAllWindows()