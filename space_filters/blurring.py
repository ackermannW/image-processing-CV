import cv2 as cv

img = cv.imread('m1a2_abrams_l5.jpg')
cv.imshow('Original', img)

# Avergaing filter
average = cv.blur(img, (3,3))
cv.imshow('Average blur', average)

# Gaussian blur
gaussian = cv.GaussianBlur(img, (3,3), 0)
cv.imshow('Gaussian blur', gaussian)

# Median blur, good for salt and pepper noise
median = cv.medianBlur(img, 3)
cv.imshow('Median', median)

# Bilater blurring, applies blurring but retains edges in the image
bilateral = cv.bilateralFilter(img, 10, 35, 25)
cv.imshow('Bilateral filter', bilateral)

cv.waitKey(0)
