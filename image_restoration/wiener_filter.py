import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2 as cv


def main():
    script_path = os.path.abspath(sys.argv[0])
    path = os.path.abspath(os.path.join(os.path.dirname(script_path), '..', 'images', 'lena.jpg'))

    img = cv.imread(path, 0)
    img = cv.GaussianBlur(img, (11,11), 0)
    G = np.fft.fftshift(np.fft.fft2(img))
    K = 10

    h = cv.getGaussianKernel(3, 0)
    h = np.dot(h, h.T)
    #h = h/np.sum(h)
    H = np.fft.fftshift(np.fft.fft2(h, s = img.shape))
    filter = np.conj(H)/(np.abs(H)**2+K)

    F_est = filter*G
    F_est_shift = np.fft.ifftshift(F_est)
    f_est = np.abs(np.fft.ifft2(F_est_shift))

    fig, axs = plt.subplots(2,3)
    axs[0,0].imshow(img, cmap='gray')

    axs[0,1].imshow(np.log1p(np.abs(G)), cmap='gray')

    axs[0,2].imshow(np.log1p(np.abs(H)), cmap='gray')
    axs[1,0].imshow(np.log1p(np.abs(filter)), cmap='gray')
    axs[1,1].imshow(np.log1p(np.abs(F_est)), cmap='gray')

    axs[1,2].imshow(f_est, cmap='gray')

    for axis in axs.flat:
        axis.axis('off')

    fig.show()
    plt.show()

if __name__ == '__main__':
    main()