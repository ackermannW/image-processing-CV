import cv2 as cv
import numpy as np

image = cv.imread('../lena.jpg')
M = np.ones(image.shape, dtype="uint8") * 100
added = cv.add(image, M)
cv.imshow("Added", added)

M = np.ones(image.shape, dtype="uint8") * 50
subtracted = cv.subtract(image, M)
cv.imshow("Subtracted", subtracted)

cv.waitKey(0)
