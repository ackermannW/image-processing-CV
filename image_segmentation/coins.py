import os
import numpy as np
import cv2 as cv
import sys
import matplotlib.pyplot as plt

def main():
    script_path = os.path.abspath(sys.argv[0])
    path = os.path.abspath(os.path.join(os.path.dirname(script_path), '..', 'images', 'coins.jpg'))
    img = cv.imread(path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply watershed algorithm
    markers = cv.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    # Plotting all images in one screen
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    axs[0, 0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    axs[0, 0].set_title('Segmented Image')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(thresh, cmap='gray')
    axs[0, 1].set_title('Binary Image')
    axs[0, 1].axis('off')

    axs[0, 2].imshow(opening, cmap='gray')
    axs[0, 2].set_title('Opened Image')
    axs[0, 2].axis('off')

    axs[1, 0].imshow(sure_bg, cmap='gray')
    axs[1, 0].set_title('Sure Background')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(sure_fg, cmap='gray')
    axs[1, 1].set_title('Sure Foreground')
    axs[1, 1].axis('off')

    axs[1, 2].imshow(unknown, cmap='gray')
    axs[1, 2].set_title('Unknown')
    axs[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
