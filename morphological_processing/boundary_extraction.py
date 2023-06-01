# Import the necessary libraries
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    path = os.path.join('..', os.getcwd(), 'images', 'coins.jpg')
    img  = cv.imread(path, flags=0) # flag 0 for grayscale image
    # Read the image as a grayscale image
    cv.imshow('image', img)
    cv.waitKey(0)
    
    # Threshold the image
    ret,img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    #img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3) 
    cv.imshow('binarized', img)
    cv.waitKey(0)

    # Get a Cross Shaped Kernel
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
    #Step 1: Dilate the image
    open = cv.morphologyEx(img, cv.MORPH_ERODE, element)
    #Step 3: Substract dilated from the original image
    boundary = cv.subtract(img, open)
    
    # Displaying the final boundary
    cv.imshow("Boundary",boundary)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()