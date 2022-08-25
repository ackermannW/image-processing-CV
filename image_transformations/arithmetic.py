import cv2 as cv
import numpy as np
import os

def main():
    path = os.path.join('..', os.getcwd(), 'images', 'lena.jpg')

    image = cv.imread(path)
    M = np.ones(image.shape, dtype="uint8") * 100
    added = cv.add(image, M)
    cv.imshow("Added", added)

    M = np.ones(image.shape, dtype="uint8") * 50
    subtracted = cv.subtract(image, M)
    cv.imshow("Subtracted", subtracted)

    cv.waitKey(0)

if __name__ == '__main__':
    main()
