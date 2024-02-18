import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import sys

def main():
    script_path = os.path.abspath(sys.argv[0])
    path = os.path.abspath(os.path.join(os.path.dirname(script_path), '..', 'images', 'lena.jpg'))
    
    img = cv.imread(path, 0)
    hist, _ = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()

    equ = cv.equalizeHist(img)
    hist_corrected, _ = np.histogram(equ.flatten(),256,[0,256])
    cdf_corrected = hist_corrected.cumsum()
    cdf_corrected_normalized = cdf_corrected * float(hist_corrected.max()) / cdf_corrected.max()

    plt.subplot(2,2,3)
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    
    plt.subplot(2,2,4)
    plt.plot(cdf_corrected_normalized, color='b')
    plt.hist(equ.flatten(), 256, [0,256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf','histogram'), loc = 'upper left')

    plt.subplot(2,2,1)
    plt.imshow(img, cmap='gray')
    plt.title('Original image')
    plt.axis('off')

    plt.subplot(2,2,2)
    plt.imshow(equ, cmap='gray')
    plt.title('Equalized histogram')
    plt.axis('off')
    
    plt.show()
if __name__=='__main__':
    main()