import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys

def main():
    script_path = os.path.abspath(sys.argv[0])
    path = os.path.abspath(os.path.join(os.path.dirname(script_path), '..', 'images', 'coins.jpg'))
    img  = cv.imread(path)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    cv.imshow('binary image', thresh)
    cv.waitKey(0)

    # noise removal
    # we use opening to remove smal white noises from the image
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)

    cv.imshow('opened image', opening)
    cv.waitKey(0)

    # sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    
    cv.imshow('Distance transform', dist_transform)
    cv.waitKey(0)
    
    ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)

    cv.imshow('Sure foreground', sure_fg)
    cv.waitKey(0)

    cv.imshow('Unknown', unknown)
    cv.waitKey(0)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    
    markers = cv.watershed(img,markers)
    img[markers == -1] = [255,0,0]
    
    cv.imshow('segmentated image',img)
    cv.waitKey(0)
if __name__ == '__main__':
    main()