import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import sys
def notch_reject_filter(shape, d0=9, u_k=0, v_k=0):
    P, Q = shape
    # Initialize filter with zeros
    H = np.zeros((P, Q))

    # Traverse through filter
    for u in range(0, P):
        for v in range(0, Q):
            # Get euclidean distance from point D(u,v) to the center
            D_uv = np.sqrt((u - P / 2 + u_k) ** 2 + (v - Q / 2 + v_k) ** 2)
            D_muv = np.sqrt((u - P / 2 - u_k) ** 2 + (v - Q / 2 - v_k) ** 2)

            H[u,v] = (1/(1+(d0/D_uv)**4))*(1/(1+(d0/D_muv)**4))

    return H

def main():
    script_path = os.path.abspath(sys.argv[0])
    path = os.path.abspath(os.path.join(os.path.dirname(script_path), '..', '..', 'images', 'moire.png'))

    imgGray  = cv.imread(path, flags=0) # flags=0 to import as grrayscale

    f = np.fft.fft2(imgGray)
    f_shift = np.fft.fftshift(f)

    M,N = f.shape
    
    H1 = notch_reject_filter((M,N), 9, 38, 30)
    H2 = notch_reject_filter((M,N), 9, -42, 27)
    H3 = notch_reject_filter((M,N), 9, 80, 30)
    H4 = notch_reject_filter((M,N), 9, -82, 28)

    H = H1*H2*H3*H4
    G_shift = f_shift * H
    G = np.fft.ifftshift(G_shift)
    
    g = np.abs(np.fft.ifft2(G))
    spectrum = np.log1p(np.abs(f_shift))
    indices = np.where(spectrum == spectrum.max())
    print(indices)
    fig, axs = plt.subplots(2,3)
    axs[0,0].imshow(imgGray, cmap='gray')
    axs[0,0].set_title('Original image')

    axs[0,1].imshow(np.log1p(np.abs(f_shift)), cmap='gray')
    axs[0,1].set_title('Magnitude spectrum')

    axs[0,2].imshow(H, cmap='gray')
    axs[0,2].set_title('Notch filter')

    axs[1,0].imshow(np.log1p(np.abs(G_shift)), cmap='gray')
    axs[1,0].set_title('Centered filtered FT')

    axs[1,1].imshow(np.log1p(np.abs(G)), cmap='gray')
    axs[1,1].set_title('Shifted filtered FT')

    axs[1,2].imshow(g, cmap='gray')
    axs[1,2].set_title('Restored image')

    fig.show()

    plt.tight_layout()

    plt.show()
if __name__ == '__main__':
    main()
