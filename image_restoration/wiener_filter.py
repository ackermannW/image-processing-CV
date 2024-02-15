import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2 as cv

def degradation_function(M,N, k = 0.0025):
    H = np.zeros((M,N), dtype=np.float32)
    for u in range(M):
        for v in range(N):
            H[u, v] = np.exp(-k*((u - M/2)**2 + (v-N/2)**2)**(5/6))
    
    return H

def degrade_image(img, k = 0.0025):
    M,N = img.shape
    H = np.zeros((M,N), dtype=np.float32)
    for u in range(M):
        for v in range(N):
            H[u, v] = np.exp(-k*((u - M/2)**2 + (v-N/2)**2)**(5/6))
    
    F = np.fft.fftshift(np.fft.fft2(img))
    F_degraded = F*H
    return np.abs(np.fft.ifft2(np.fft.ifftshift(F_degraded)))

def mse(img, degraded_img):
    return np.square(np.subtract(img, degraded_img)).mean()

def main():
    script_path = os.path.abspath(sys.argv[0])
    path = os.path.abspath(os.path.join(os.path.dirname(script_path), '..', 'images', 'city.png'))

    img = cv.imread(path, 0)
    degraded_image = degrade_image(img)
    G = np.fft.fftshift(np.fft.fft2(degraded_image))
    K = 0.03

    h = cv.getGaussianKernel(3, 0)
    h = np.dot(h, h.T)
    H = degradation_function(M=img.shape[0], N=img.shape[1])
    wiener_filter = np.conj(H)/(np.abs(H)**2+K)

    F_est = wiener_filter*G
    F_est_shift = np.fft.ifftshift(F_est)
    f_est = np.abs(np.fft.ifft2(F_est_shift))

    print(f'MSE for degraded image = {mse(img, degraded_image)}')
    print(f'MSE for Wiener filter = {mse(f_est, degraded_image)}')

    fig, axs = plt.subplots(1,3)
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original image')

    axs[1].imshow(degraded_image, cmap='gray')
    axs[1].set_title('Degraded image')

    axs[2].imshow(f_est, cmap='gray')
    axs[2].set_title('Restored image')

    for axis in axs.flat:
        axis.axis('off')

    fig.show()
    plt.show()

if __name__ == '__main__':
    main()