import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import sys

def main():
    script_path = os.path.abspath(sys.argv[0])
    path = os.path.abspath(os.path.join(os.path.dirname(script_path), '..', '..', 'images', 'lena.jpg'))
    img = cv.imread(path, 0)

    f = np.fft.fftshift(np.fft.fft2(img))

    M,N = f.shape
    H = np.zeros((M,N), dtype=np.float32)
    D0 = 10
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - M/2)**2+(v-N/2)**2)
            H[u,v] = np.exp(-D**2/(2*D0*D0))

    FLP = H*f

    FLP = np.fft.ifftshift(FLP)
    flp = np.abs(np.fft.ifft2(FLP))

    g_mask = img - flp

    k = 1 # > 1 is highboost filtering 
    g = img + k*g_mask
    g = np.clip(g, 0, 255)

    fig, axs = plt.subplots(2,3)
    axs[0,0].imshow(img, cmap='gray')
    axs[0,0].set_title('Original image')

    axs[0,1].imshow(np.log1p(np.abs(f)), cmap='gray')
    axs[0,1].set_title('Fourier spectrum')

    axs[0,2].imshow(H, cmap='gray')
    axs[0,2].set_title('Gaussian filter')

    axs[1,0].imshow(flp, cmap='gray')
    axs[1,0].set_title('Smoothed image')

    axs[1,1].imshow(g_mask, cmap='gray')
    axs[1,1].set_title('Mask')
    
    axs[1,2].imshow(g, cmap='gray')
    axs[1,2].set_title('Sharpened image')
    
    fig.show()
    plt.show()
    
if __name__ == '__main__':
    main()