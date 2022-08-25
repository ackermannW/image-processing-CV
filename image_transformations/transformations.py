import utils
import cv2 as cv

image = cv.imread('../lena.jpg')
shifted = utils.translate(image, 200, 100)
cv.imshow('Shifted', shifted)

rotated = utils.rotate(image, 45)
cv.imshow('Rotated', rotated)

resized = utils.resize(image, 400)
cv.imshow('Resized', resized)

# 1 for horizontal flip
# 0 for vertical flip
# -1 for both
flipped = cv.flip(image, 1)
cv.imshow('Flipped', flipped)

cropped = image[30:100, 240:300]
cv.imshow('Cropped', cropped)

cv.waitKey(0)
