import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import sys

def main():
    script_path = os.path.abspath(sys.argv[0])
    path = os.path.abspath(os.path.join(os.path.dirname(script_path), '..', '..', 'images', 'lena.jpg'))

    imgGray  = cv.imread(path, flags=0) # flags=0 to import as grrayscale

    f = np.fft.fft2(imgGray)
    f_shift = np.fft.fftshift(f)

    M,N = f.shape
    H = np.zeros((M,N), dtype=np.float32)
    D0 = 100
    
    H = np.zeros(shape=f.shape, dtype = "uint8")
    cv.circle(H, (M//2, N//2), D0, (255,255,255), -1)

    G_shift = f_shift * H
    G = np.fft.ifftshift(G_shift)
    
    g = np.abs(np.fft.ifft2(G))

    fig, axs = plt.subplots(2,3)
    axs[0,0].imshow(imgGray, cmap='gray')
    axs[0,0].set_title('Original image')

    axs[0,1].imshow(np.log1p(np.abs(f_shift)), cmap='gray')
    axs[0,1].set_title('Magnitude spectrum')

    axs[0,2].imshow(H, cmap='gray')
    axs[0,2].set_title('Gaussian filter')

    axs[1,0].imshow(np.log1p(np.abs(G_shift)), cmap='gray')
    axs[1,0].set_title('Centered filtered FT')

    axs[1,1].imshow(np.log1p(np.abs(G)), cmap='gray')
    axs[1,1].set_title('Shifted filtered FT')

    axs[1,2].imshow(g, cmap='gray')
    axs[1,2].set_title('Resotred image')

    fig.show()

    plt.tight_layout()

    plt.show()
if __name__ == '__main__':
    main()
