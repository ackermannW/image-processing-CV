import cv2 as cv
import numpy as np
import os

def main():
    path = os.path.join('..', os.getcwd(), 'images', 'm1a2_abrams_l5.jpg')
    img = cv.imread(path)
    cv.imshow('Original', img)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Abrams Gray', gray)

    # Labplacian filter
    lap = cv.Laplacian(gray, cv.CV_64F)
    lap = np.uint8(np.absolute(lap))
    cv.imshow('Laplacian', lap)

    # Sobel filter
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0 )
    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
    combined_sobel = cv.bitwise_or(sobelx, sobely)

    cv.imshow('Sobelx', sobelx)
    cv.imshow('Sobely', sobely)
    cv.imshow('Combined Sobel', combined_sobel)

    canny = cv.Canny(gray, 150,175)
    cv.imshow('Canny', canny)

    cv.waitKey(0)

if __name__ == '__main__':
    main()