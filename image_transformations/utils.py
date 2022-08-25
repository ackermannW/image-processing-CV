import numpy as np
import cv2 as cv


def translate(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted


def rotate(image, angle, rot_point=None, scale = 1.0):
    (height, width) = image.shape[:2]
    if rot_point is not None:
        rot_point = (width // 2, height // 2)  # // operator denotes integer division
    rot_matrix = cv.getRotationMatrix2D(rot_point, angle, scale)
    rotated = cv.warpAffine(image, rot_matrix, (width, height))
    return rotated


def resize(image, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(h)
        dim = (width, int(h * r))

    resized = cv.resize(image, dim, interpolation=inter)
    return resized
