import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import sys

def main():
    script_path = os.path.abspath(sys.argv[0])
    path = os.path.abspath(os.path.join(os.path.dirname(script_path), '..', 'images', 'aerial.png'))
    
    
    imgGray = cv.imread(path, 0)
    c = 1

    plt.subplot(2,2,1)
    plt.imshow(imgGray, cmap='gray')
    plt.title('Original image')
    plt.axis('off')

    for i, gamma in enumerate([3,4,5]):
        s = c*np.array(255*(imgGray / 255) ** gamma, dtype='uint8')

        plt.subplot(2,2,i+2)
        plt.imshow(s, cmap='gray')
        plt.title(f'gamma = {gamma}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
if __name__=='__main__':
    main()