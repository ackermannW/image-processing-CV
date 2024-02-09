import numpy as np
import cv2 as cv
import os
import sys

def main():
    script_path = os.path.abspath(sys.argv[0])
    path = os.path.abspath(os.path.join(os.path.dirname(script_path), '..', 'images', 'squares.jpg'))
    image  = cv.imread(path)
    cv.imshow('input image', image)
    cv.waitKey(0)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edged = cv.Canny(gray, 30, 200)
    cv.imshow('canny edges', edged)
    cv.waitKey(0)
    contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cv.imshow('edges after contouring', edged)
    cv.waitKey(0)
    print(contours)
    print('Number of contours found=' + str(len(contours)))

    #use -1 as the 3rd parameter to draw all the contours
    cv.drawContours(image,contours,-1,(0,255,0),3)
    cv.imshow('contours',image)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()