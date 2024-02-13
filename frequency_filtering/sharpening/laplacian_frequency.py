import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import sys

def main():
    script_path = os.path.abspath(sys.argv[0])
    path = os.path.abspath(os.path.join(os.path.dirname(script_path), '..', '..', 'images', 'moon.png'))
    imgGray = cv.imread(path, 0)/255
    f = np.fft.fft2(imgGray)
    f_shift = np.fft.fftshift(f)

    M,N = f.shape
    H = np.zeros((M,N), dtype=np.float32)

    for u in range(M):
        for v in range(N):
            H[u,v] = -4*np.pi*np.pi*((u-M/2)**2 + (v-N/2)**2)

    G_shift = f_shift * H
    G = np.fft.ifftshift(G_shift)
    g = np.abs(np.fft.ifft2(G))
    g = g / np.max(g)
    
    c = -1
    sharpened_image = imgGray + c*g
    
    sharpened_image = np.clip(sharpened_image, 0,1)*255

    fig, axs = plt.subplots(2,3)

    axs[0,0].imshow(imgGray, cmap='gray')
    axs[0,0].set_title('Original image')

    axs[0,1].imshow(np.log1p(np.abs(f_shift)), cmap='gray')
    axs[0,1].set_title('Image Foruier spectrum')

    axs[0,2].imshow(H, cmap='gray')
    axs[0,2].set_title('Laplacian filter')

    axs[1,0].imshow(np.log1p(np.abs(G_shift)), cmap='gray')
    axs[1,0].set_title('Filtered spectrum')

    axs[1,1].imshow(g, cmap='gray')
    axs[1,1].set_title('Mask')

    axs[1,2].imshow(sharpened_image, cmap='gray')
    axs[1,2].set_title('Sharpened image')

    fig.show()
    plt.tight_layout()
    plt.show()
if __name__=='__main__':
    main()