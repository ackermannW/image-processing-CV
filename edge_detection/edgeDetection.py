import cv2 as cv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

def main():
    script_path = os.path.abspath(sys.argv[0])
    path = os.path.abspath(os.path.join(os.path.dirname(script_path), '..', 'images', 'm1a2_abrams_l5.jpg'))
    img = cv.imread(path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    lap = cv.Laplacian(gray, cv.CV_64F)
    lap = np.uint8(np.absolute(lap))
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
    combined_sobel = cv.bitwise_or(np.uint8(np.absolute(sobelx)), np.uint8(np.absolute(sobely)))
    canny = cv.Canny(gray, 150, 175)
    sharpened_lap = cv.subtract(gray, lap)
    sharpened_sobel = cv.subtract(gray, combined_sobel.astype(np.uint8))
    sharpened_canny = cv.subtract(gray, canny)

    fig, axs = plt.subplots(2, 4, figsize=(14, 7))

    axs[0, 0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    axs[0, 0].set_title('Original')

    axs[0, 1].imshow(gray, cmap='gray')
    axs[0, 1].set_title('Grayscale')

    axs[0, 2].imshow(lap, cmap='gray')
    axs[0, 2].set_title('Laplacian')

    axs[0, 3].imshow(combined_sobel, cmap='gray')
    axs[0, 3].set_title('Combined Sobel')

    axs[1, 0].imshow(sobelx, cmap='gray')
    axs[1, 0].set_title('Sobel X')

    axs[1, 1].imshow(sobely, cmap='gray')
    axs[1, 1].set_title('Sobel Y')

    axs[1, 2].imshow(canny, cmap='gray')
    axs[1, 2].set_title('Canny')

    axs[1, 3].imshow(sharpened_lap, cmap='gray')
    axs[1, 3].set_title('Sharpening using Laplacian')

    for ax in axs.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
