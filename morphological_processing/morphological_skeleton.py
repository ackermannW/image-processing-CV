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
    # Step 1: Create an empty skeleton
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    # Get a Cross Shaped Kernel
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
    # Repeat steps 2-4
    while True:
        #Step 2: Open the image
        open = cv.morphologyEx(img, cv.MORPH_OPEN, element)
        #Step 3: Substract open from the original image
        temp = cv.subtract(img, open)
        #Step 4: Erode the original image and refine the skeleton
        eroded = cv.erode(img, element)
        skel = cv.bitwise_or(skel,temp)
        img = eroded.copy()
        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv.countNonZero(img)==0:
            break

    # Displaying the final skeleton
    cv.imshow("Skeleton",skel)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
    