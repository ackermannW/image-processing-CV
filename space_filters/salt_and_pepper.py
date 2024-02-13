import numpy as np
import cv2 as cv
import os
import numpy as np
import sys
import matplotlib.pyplot as plt

def main():
    script_path = os.path.abspath(sys.argv[0])
    path = os.path.abspath(os.path.join(os.path.dirname(script_path), '..', 'images', 'lena.jpg'))
    img = cv.imread(path, flags=0)

    pepper = 0.05
    salt = 1 - pepper
    M,N = img.shape
    noisy = np.zeros((M,N), dtype='uint8')
    for i in range(M):
        for j in range(N):
            rdn = np.random.random()
            if rdn < pepper:
                noisy[i][j] = 0
            elif rdn > salt:
                noisy[i][j] = 255
            else:
                noisy[i][j] = img[i][j]

    filtered = cv.medianBlur(noisy, 3)
    
    fig, axs = plt.subplots(1,3)

    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original image')
    axs[0].set_xticklabels([])
    axs[0].set_yticklabels([])

    axs[1].imshow(noisy, cmap='gray')
    axs[1].set_title('Salt and pepper')
    axs[1].set_xticklabels([])
    axs[1].set_yticklabels([])

    axs[2].imshow(filtered, cmap='gray')
    axs[2].set_title('Median filtered')
    axs[2].set_xticklabels([])
    axs[2].set_yticklabels([])

    fig.show()
    plt.show()
if __name__=='__main__':
    main()