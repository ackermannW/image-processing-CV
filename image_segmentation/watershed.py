import cv2 as cv
import os
from matplotlib import pyplot as plt
import numpy as np

def main():


    path = os.path.join('..', os.getcwd(), 'images', 'coins.jpg')
    img = cv.imread(path)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

    plt.subplot(3,3,1)
    plt.title('Original')
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

    plt.subplot(3,3,2)
    plt.title('Binary')
    plt.imshow(thresh, cmap='gray')

    # to remove white noise we use morph opening
    # to remove small holes in the objects we can use morph closing

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)

    # dilation increases object boundary 
    # whatever remains is surely background
    # sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3)

    # erosion removes the boundary pixels, that way we will find parts of image which surely 
    # contain the foreground 
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    plt.subplot(3,3,3)
    plt.title('Sure background')
    plt.imshow(sure_bg, cmap='gray')
    
    plt.subplot(3,3,4)
    plt.title('Distance transform')
    plt.imshow(dist_transform, cmap='gray')

    plt.subplot(3,3,5)
    plt.title('Sure forground')
    plt.imshow(sure_fg, cmap='gray')

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    plt.subplot(3,3,6)
    plt.title('Markers')
    plt.imshow(markers, cmap='jet')

    markers = cv.watershed(img,markers)
    img[markers == -1] = [255,0,0]

    plt.show()

if __name__=='__main__':
    main()