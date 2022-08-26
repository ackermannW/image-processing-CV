import numpy as np
import cv2 as cv

def gaussLowPassFilter(shape, radius=10):  # Gaussian low pass filter
    # Gaussian filter:# Gauss = 1/(2*pi*s2) * exp(-(x**2+y**2)/(2*s2))
    u, v = np.mgrid[-1:1:2.0/shape[0], -1:1:2.0/shape[1]]
    D = np.sqrt(u**2 + v**2)
    D0 = radius / shape[0]
    kernel = np.exp(- (D ** 2) / (2 * D0**2))
    return kernel

def gaussHighPassFilter(shape, radius=10):
    kernel = 1 - gaussLowPassFilter(shape, radius)
    return kernel

def butterWorthLowPassFilter(shape, radius=10):
    # Butterworth filter:# Gauss = 1/(2*pi*s2) * exp(-(x**2+y**2)/(2*s2))
    u, v = np.mgrid[-1:1:2.0/shape[0], -1:1:2.0/shape[1]]
    D = np.sqrt(u**2 + v**2)
    D0 = radius / shape[0]
    kernel = 1/(1 + ((D ** 2) / ( D0**2)))
    return kernel

def butterWorthHighPassFilter(shape, radius=10):
    kernel = 1 - butterWorthLowPassFilter(shape, radius)
    return kernel

def idealLowPassFilter(shape, radius=10):
    m = shape[0]
    n = shape[1]
    blank = np.zeros((m,n), np.uint8)
    xc = n//2
    yc = n//2
    blank[xc-radius:xc+radius, yc-radius:yc+radius] = 1
    return blank

def idealHighPassFilter(shape, radius=10):
    m = shape[0]
    n = shape[1]
    blank = np.ones((m,n), np.uint8)
    xc = n//2
    yc = n//2
    blank[xc-radius:xc+radius, yc-radius:yc+radius] = 0
    return blank

def dft2Image(image):  # Optimal extended fast Fourier transform
    # Centralized 2D array f (x, y) * - 1 ^ (x + y)
    mask = np.ones(image.shape)
    mask[1::2, ::2] = -1
    mask[::2, 1::2] = -1
    fImage = image * mask  # f(x,y) * (-1)^(x+y)

    # Optimal DFT expansion size
    # The height and width of the original picture
    rows, cols = image.shape[:2]
    rPadded = cv.getOptimalDFTSize(rows)  # Optimal DFT expansion size
    cPadded = cv.getOptimalDFTSize(cols)  # For fast Fourier transform

    # Edge extension (complement 0), fast Fourier transform
    # Edge expansion of the original image
    dftImage = np.zeros((rPadded, cPadded, 2), np.float32)
    # Edge expansion, 0 on the lower and right sides
    dftImage[:rows, :cols, 0] = fImage
    # fast Fourier transform
    cv.dft(dftImage, dftImage, cv.DFT_COMPLEX_OUTPUT)
    return dftImage
