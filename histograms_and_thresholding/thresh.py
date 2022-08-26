import cv2 as cv
from cv2 import threshold
import matplotlib.pyplot as plt
from numpy import histogram
import os

def main():
    path = os.path.join('..', os.getcwd(), 'images', 'lena.jpg')
    img = cv.imread(path)

    cv.imshow('Original', img)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Grayscale', gray)

    gray_hist = cv.calcHist([img], [0], None, [256], [0, 256])

    plt.figure()
    plt.title('Grayscale histogram')
    plt.xlabel('Bins')
    plt.ylabel('Nummber of pixels')
    plt.plot(gray_hist)
    plt.xlim([0,256])

    plt.show()

    threshold, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
    cv.imshow('Thresholded', thresh)

    threshold, thresh_inv = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)
    cv.imshow('Thresholded inverse', thresh_inv)

    adaptive_tresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)
    cv.imshow('Adaptive thresholding', adaptive_tresh)

    cv.waitKey(0)

if __name__ == '__main__':
    main()