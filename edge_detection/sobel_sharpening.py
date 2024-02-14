import cv2 as cv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

def main():
    script_path = os.path.abspath(sys.argv[0])
    path = os.path.abspath(os.path.join(os.path.dirname(script_path), '..', 'images', 'moon.png'))
    img = cv.imread(path, 0)

    sobelx = cv.Sobel(img, cv.CV_64F, 1, 0)
    sobely = cv.Sobel(img, cv.CV_64F, 0, 1)
    #combined_sobel = cv.bitwise_or(np.uint8(np.absolute(sobelx)), np.uint8(np.absolute(sobely)))
    combined_sobel = np.sqrt(np.square(sobelx)**0.5 + np.square(sobely)**0.5)
    c = 1 
    g_x = img + c*sobelx
    g_x = np.clip(g_x, 0, 255)

    g_y = img + c*sobely
    g_y = np.clip(g_y, 0, 255)
    
    g = img + c*combined_sobel
    g = np.clip(g, 0, 255)

    fig, axs = plt.subplots(2,3)

    axs[0,0].imshow(img, cmap='gray')
    axs[0,0].set_title('Original')

    axs[0,1].imshow(sobelx, cmap='gray')
    axs[0,1].set_title('Sobel X')

    axs[0,2].imshow(sobely, cmap='gray')
    axs[0,2].set_title('Sobel Y')

    axs[1,0].imshow(combined_sobel, cmap='gray')
    axs[1,0].set_title('Gradient')

    axs[1,1].imshow(g_x, cmap='gray')
    axs[1,1].set_title('Enhanced horizontal edges')

    axs[1,2].imshow(g_y, cmap='gray')
    axs[1,2].set_title('Enhanced vertical edges')

    for axis in axs.flat:
        axis.axis('off')

    plt.tight_layout()
    fig.show()
    plt.show()

    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title('Original image')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(g, cmap='gray')
    plt.title('Sharepened image')
    plt.axis('off')

    plt.show()
if __name__ == '__main__':
    main()
