from operator import countOf
import numpy as np
import cv2 as cv
import os

def main():
    path = os.path.join('..', os.getcwd(), 'images', 'house.jpg')
    image  = cv.imread(path)
    orig_image=image.copy()
    cv.imshow('original image',orig_image)
    cv.waitKey(0)
    gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret, thresh=cv.threshold(gray,127,255,cv.THRESH_BINARY_INV)

    contours, hierarchy=cv.findContours(thresh.copy(),cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
    for c in contours:
        x,y,w,h=cv.boundingRect(c)
        cv.rectangle(orig_image,(x,y),(x+w,y+h),(0,0,255),2)
        cv.imshow('Bounding rect',orig_image)
    cv.waitKey(0)

    for c in contours:
    #calculate accuracy as a percent of contour perimeter
        accuracy=0.03*cv.arcLength(c,True)
        approx=cv.approxPolyDP(c,accuracy,True)
        cv.drawContours(image,[approx],0,(0,255,0),2)
        cv.imshow('Approx polyDP', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()