# Import the necessary libraries
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

def main():
    script_path = os.path.abspath(sys.argv[0])
    path = os.path.abspath(os.path.join(os.path.dirname(script_path), '..', 'images', 'coins.jpg'))
    img  = cv.imread(path) # flag 0 for grayscale image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Read the image as a grayscale image
    cv.imshow('image', gray)
    cv.waitKey(0)
    
    # Threshold the image
    ret,binarized = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    #img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3) 
    cv.imshow('binarized', binarized)
    cv.waitKey(0)

    # Get a Cross Shaped Kernel
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
    #Step 1: Dilate the image
    open = cv.morphologyEx(binarized, cv.MORPH_ERODE, element)
    #Step 3: Substract dilated from the original image
    boundary = cv.subtract(binarized, open)
    boundary = cv.dilate(boundary, kernel=np.ones((3,3), np.uint8), iterations=1)

    boundary_color = cv.cvtColor(boundary, cv.COLOR_GRAY2BGR)
    boundary_color[:,:,0] = 0
    boundary_color[:,:,1] = 0

    img_with_boundary = cv.add(img, boundary_color)

    # Displaying the final boundary
    cv.imshow("Boundary",boundary)
    cv.imshow("Final image", img_with_boundary)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
    