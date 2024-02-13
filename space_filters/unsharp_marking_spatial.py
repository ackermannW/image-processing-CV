import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import sys

def main():
    script_path = os.path.abspath(sys.argv[0])
    path = os.path.abspath(os.path.join(os.path.dirname(script_path), '..', 'images', 'lena.jpg'))
    f = cv.imread(path, 0) / 255

    f_blur = cv.GaussianBlur(src=f, ksize=(31,31), sigmaX=0, sigmaY=0)
    g_mask = f - f_blur

    k = 1
    g = f + g_mask
    g = np.clip(g, 0, 1)
    
    fig, axs = plt.subplots(2,2)
    axs[0,0].imshow(f, cmap='gray')
    axs[0,0].set_title('Original image')

    axs[0,1].imshow(f_blur, cmap='gray')
    axs[0,1].set_title('Blurred')

    axs[1,0].imshow(g_mask, cmap='gray')
    axs[1,0].set_title('Mask')

    axs[1,1].imshow(g, cmap='gray')
    axs[1,1].set_title('Sharpened image')
    
    fig.show()
    plt.show()
    
if __name__ == '__main__':
    main()