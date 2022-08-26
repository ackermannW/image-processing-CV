import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import filter_utilities as utils

def main():
    path = os.path.join('..', os.getcwd(), 'images', 'lena.jpg')
    imgGray  = cv.imread(path, flags=0) # flags=0 to import as grrayscale
    rows, cols = imgGray.shape[:2]  # The height and width of the picture
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 3, 1),plt.title("Original"), plt.axis('off'), plt.imshow(imgGray, cmap='gray')

     # (2) Fast Fourier transform
    dftImage = utils.dft2Image(imgGray)  # Fast Fourier transform (rPad, cPad, 2)
    rPadded, cPadded = dftImage.shape[:2]  # Fast Fourier transform size, original image size optimization
    print("dftImage.shape:{}".format(dftImage.shape))

    D0 = [10, 30, 60, 90, 120]  # radius
    for k in range(5):
        # (3) Construct Gaussian low pass filter
        lpFilter = utils.gaussLowPassFilter((rPadded, cPadded), radius=D0[k])

        # (5) Modify Fourier transform in frequency domain: Fourier transform point multiplication low-pass filter
        dftLPfilter = np.zeros(dftImage.shape, dftImage.dtype)  # Size of fast Fourier transform (optimized size)
        for j in range(2):
            dftLPfilter[:rPadded, :cPadded, j] = dftImage[:rPadded, :cPadded, j] * lpFilter

        # (6) The inverse Fourier transform is performed on the low-pass Fourier transform, and only the real part is taken
        idft = np.zeros(dftImage.shape[:2], np.float32)  # Size of fast Fourier transform (optimized size)
        cv.dft(dftLPfilter, idft, cv.DFT_REAL_OUTPUT + cv.DFT_INVERSE + cv.DFT_SCALE)

        # (7) Centralized 2D array g (x, y) * - 1 ^ (x + y)
        mask2 = np.ones(dftImage.shape[:2])
        mask2[1::2, ::2] = -1
        mask2[::2, 1::2] = -1
        idftCen = idft * mask2  # g(x,y) * (-1)^(x+y)

        # (8) Intercept the upper left corner, the size is equal to the input image
        result = np.clip(idftCen, 0, 255)  # Truncation function, limiting the value to [0255]
        imgLPF = result.astype(np.uint8)
        imgLPF = imgLPF[:rows, :cols]

        plt.subplot(2,3,k+2), plt.title("GLPF rebuild(n={})".format(D0[k])), plt.axis('off')
        plt.imshow(imgLPF, cmap='gray')

    print("image.shape:{}".format(imgGray.shape))
    print("lpFilter.shape:{}".format(lpFilter.shape))
    print("dftImage.shape:{}".format(dftImage.shape))

    plt.tight_layout()

    plt.show()
    # cv.waitKey(0)
if __name__ == '__main__':
    main()
