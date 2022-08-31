import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def main():
    path = os.path.join('..', os.getcwd(), 'images', 'coins.jpg')
    img  = cv.imread(path, flags=0) # flag 0 for grayscale image
    ret, thresh = cv.threshold(img, 100, 255, cv.THRESH_BINARY_INV)
    #thresh = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)
    kernel = np.ones((5,5), np.uint8)
    img_erosion = cv.erode(thresh, kernel, iterations=1)
    img_dilation = cv.dilate(thresh, kernel, iterations=1)
    img_opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    img_closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

    plt.subplot(2,3,1)
    plt.axis('off')
    plt.title('Original')
    plt.imshow(img, cmap='gray')

    plt.subplot(2,3,2)
    plt.axis('off')
    plt.title('Binary')
    plt.imshow(thresh, cmap='binary')

    plt.subplot(2,3,3)
    plt.axis('off')
    plt.title('Erosion')
    plt.imshow(img_erosion, cmap='binary')

    plt.subplot(2,3,4)
    plt.axis('off')
    plt.title('Dilation')
    plt.imshow(img_dilation, cmap='binary')

    plt.subplot(2,3,5)
    plt.axis('off')
    plt.title('Opening')
    plt.imshow(img_opening, cmap='binary')

    plt.subplot(2,3,6)
    plt.axis('off')
    plt.title('Closing')
    plt.imshow(img_closing, cmap='binary')

    plt.show()

if __name__ == '__main__':
    main()