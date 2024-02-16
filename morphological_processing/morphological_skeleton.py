import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

def main():
    script_path = os.path.abspath(sys.argv[0])
    path = os.path.abspath(os.path.join(os.path.dirname(script_path), '..', 'images', 'coins.jpg'))
    img  = cv.imread(path, flags=0) # flag 0 for grayscale image
    
    # Threshold the image
    _, bin_img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    
    plt.subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    plt.title('Original image')
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(bin_img, cmap='gray')
    plt.title('Binarized image')
    plt.axis('off')

    # Step 1: Create an empty skeleton
    skel = np.zeros(img.shape, np.uint8)

    # Get a Cross Shaped Kernel
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
    # Repeat steps 2-4
    while True:
        #Step 2: Open the image
        open = cv.morphologyEx(bin_img, cv.MORPH_OPEN, element)
        #Step 3: Substract open from the original image
        temp = cv.subtract(bin_img, open)
        #Step 4: Erode the original image and refine the skeleton
        eroded = cv.erode(bin_img, element)
        skel = cv.bitwise_or(skel,temp)
        bin_img = eroded.copy()
        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv.countNonZero(bin_img)==0:
            break

    plt.subplot(1,3,3)
    plt.imshow(skel, cmap='gray')
    plt.title('Skeleton')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()
    