import os
import matplotlib.pyplot as plt
from Wiener_Filter import blur, gaussian_kernel, wiener_filter, add_gaussian_noise
import cv2 as cv
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

if __name__ == '__main__':
	# Load image and convert it to gray scale
    script_path = os.path.abspath(sys.argv[0])
    path = os.path.abspath(os.path.join(os.path.dirname(script_path), '..', 'images', 'lena.jpg'))

    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	# Blur image
    blurred_img = blur(img, kernel_size = 9)
	# Add Gaussian noise
    noisy_img = add_gaussian_noise(blurred_img, 20)

	# Apply Median Filter
    median_filter = cv.medianBlur(noisy_img.astype(np.uint8), 9)

	# Apply Wiener Filter
    wiener_filter = wiener_filter(noisy_img, gaussian_kernel(9), K = 0.5)

	# Display results
    fig = plt.figure(figsize = (12, 10))

    display = [img, noisy_img, median_filter, wiener_filter]
    title = ['Original image', 'Blurred + Gaussian Noise Added', 
			 'Median Filter', 'Wiener Filter']

    for i in range(len(display)):
        fig.add_subplot(2, 2, i+1)
        plt.imshow(display[i], cmap = 'gray')
        plt.title(title[i])
	
    plt.show()
