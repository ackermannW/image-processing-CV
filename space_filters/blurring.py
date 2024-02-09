import cv2 as cv
import os
import numpy as np
import sys

def noisy(noise_typ,image):
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.05
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
   elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
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
    img = cv.imread(path)
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

    # Bilater blurring, applies blurring but retains edges in the image
    bilateral = cv.bilateralFilter(img, 10, 35, 25)
    cv.imshow('Bilateral filter', bilateral)

    cv.waitKey(0)

if __name__ == '__main__':
    main()
