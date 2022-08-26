from turtle import width
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def idealLowPass(m,n,d0):
    blank = np.zeros((m,n,2), np.uint8)
    xc = n//2
    yc = n//2
    blank[m-d0:n+d0, xc-d0:yc+d0] = 1
    return blank

img = cv.imread('frequency_filtering\m1a2_abrams_l5.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

(width,height) = img.shape[:2]

filter = idealLowPass(width, height, 100)
#cv.imshow('Ideal low pass filter',np.abs(filter[[:,:]]))

dft = cv.dft(np.float32(img), flags = cv.DFT_COMPLEX_OUTPUT)

dftshift = np.fft.fftshift(dft)
#plt.imshow(np.abs(dftshift) ,cmap='gray')\
#cv.imshow('Fourier transform', dftshift)

filteredTransform = dftshift*filter
f_ishift = np.fft.ifftshift(filteredTransform)
filteredImage = cv.idft(f_ishift)
filteredImage = cv.magnitude(filteredImage[:,:,0], filteredImage[:,:,1])

plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(filteredImage, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Filtered')
plt.show()
cv.waitKey(0)
