import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import sys

def main():
    script_path = os.path.abspath(sys.argv[0])
    path = os.path.abspath(os.path.join(os.path.dirname(script_path), '..', 'images', 'lena.jpg'))
    
    r = np.arange(0, 255)
    c = 255/np.log(1+255)
    y = c*np.log(1+r)
    
    imgGray = cv.imread(path, 0)
    s = c*np.log(1+imgGray)

    plt.subplot(2,1,1)
    plt.plot(r,y)
    plt.title('Log transform')

    plt.subplot(2,2,3)
    plt.imshow(imgGray, cmap='gray')
    plt.title('Original image')
    plt.axis('off')

    plt.subplot(2,2,4)
    plt.imshow(s, cmap='gray')
    plt.title('Log transform')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
if __name__=='__main__':
    main()