from turtle import width
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def idealHighPass(m,n,d0):
    blank = np.zeros((m,n,2), np.uint8)
    xc = n//2
    yc = n//2
    blank[m-d0:n+d0, xc-d0:yc+d0] = 1
    filter = np.ones((m,n,2), np.uint8) - blank
    return filter

img = cv.imread('frequency_filtering/merkava_mk3_l4.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

(width,height) = img.shape[:2]

filter = idealHighPass(width, height, 10)

dft = cv.dft(np.float32(img), flags = cv.DFT_COMPLEX_OUTPUT)

dftshift = np.fft.fftshift(dft)
print(dftshift)
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
