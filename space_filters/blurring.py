import cv2 as cv
import os
import numpy as np
import sys

def noisy(noise_typ,image):
   if noise_typ == "gauss":
      mean = 0
      sigma = 5
      gauss = np.random.normal(mean,sigma,image.shape).astype(np.uint8)
      noisy = cv.add(image, gauss)
      return noisy
   elif noise_typ == "s&p":
      pepper = 0.05
      salt = 1 - pepper
      M,N = image.shape
      noisy = np.zeros((M,N), dtype='uint8')
      for i in range(M):
        for j in range(N):
            rdn = np.random.random()
            if rdn < pepper:
                noisy[i][j] = 0
            elif rdn > salt:
                noisy[i][j] = 255
            else:
                noisy[i][j] = image[i][j]
       
      return noisy
   elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
   elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)        
      noisy = image + image * gauss
      return noisy

def main():
    script_path = os.path.abspath(sys.argv[0])
    path = os.path.abspath(os.path.join(os.path.dirname(script_path), '..', 'images', 'lena.jpg'))
    img = cv.imread(path, 0)
    noisy_gauss_img = noisy("gauss", img)
    noisy_salt_pepper_img = noisy("s&p", img)

    cv.imshow('Original', img)
    cv.imshow('Gaussian noise', noisy_gauss_img)
    cv.imshow('Salt and pepper noise', noisy_salt_pepper_img)

    # Avergaing filter
    average = cv.blur(img, (3,3))
    cv.imshow('Average blur', average)

    # Gaussian blur
    gaussian = cv.GaussianBlur(img, (3,3), 0)
    cv.imshow('Gaussian blur', gaussian)

    # Median blur, good for salt and pepper noise
    median = cv.medianBlur(noisy_salt_pepper_img, 3)
    cv.imshow('Median', median)

    cv.waitKey(0)

if __name__ == '__main__':
    main()
