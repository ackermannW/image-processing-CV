import cv2 as cv
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

def main():
    script_path = os.path.abspath(sys.argv[0])
    path = os.path.abspath(os.path.join(os.path.dirname(script_path), '..', 'images', 'moon.png'))
    img = cv.imread(path, 0)

    kernel = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]])
    
    laplacian_image = cv.filter2D(src=img, ddepth=-1, kernel=kernel)

    c = -1 
    g = img + c*laplacian_image

    # there are negative values in g
    # therefore the enhanced image g is gray 
    # we need to clip values so that they are 
    # in range [0, 255]

    g_clip = np.clip(g, 0, 255)

    fig, axs = plt.subplots(2,2)

    axs[0,0].imshow(img, cmap='gray')
    axs[0,0].set_title('Original')

    axs[0,1].imshow(laplacian_image, cmap='gray')
    axs[0,1].set_title('Laplacian')

    axs[1,0].imshow(g, cmap='gray')
    axs[1,0].set_title('Enhanced image')

    axs[1,1].imshow(g_clip, cmap='gray')
    axs[1,1].set_title('Enhanced image')
    fig.show()
    plt.show()

if __name__ == '__main__':
    main()
