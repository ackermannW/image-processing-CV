import cv2 as cv
import os
import sys
import matplotlib.pyplot as plt

def main():
    script_path = os.path.abspath(sys.argv[0])
    path = os.path.abspath(os.path.join(os.path.dirname(script_path), '..', 'images', 'lena.jpg'))
    img = cv.imread(path, flags=0)

    filtered1 = cv.blur(img,(3,3))
    filtered2 = cv.blur(img, (5,5))
    filtered3 = cv.blur(img, (9,9))

    # also possible to do:
    # filtered = cv.GaussianBlur(img, (9,9), 0)

    fig, axs = plt.subplots(2,2)

    axs[0,0].imshow(img, cmap='gray')
    axs[0,0].set_title('Original image')
    axs[0,0].set_xticklabels([])
    axs[0,0].set_yticklabels([])

    axs[0,1].imshow(filtered1, cmap='gray')
    axs[0,1].set_title('Kernel (3,3)')
    axs[0,1].set_xticklabels([])
    axs[0,1].set_yticklabels([])

    axs[1,0].imshow(filtered1, cmap='gray')
    axs[1,0].set_title('AVG (5,5)')
    axs[1,0].set_xticklabels([])
    axs[1,0].set_yticklabels([])

    axs[0,1].imshow(filtered2, cmap='gray')
    axs[0,1].set_title('AVG (5,5)')
    axs[0,1].set_xticklabels([])
    axs[0,1].set_yticklabels([])

    axs[1,1].imshow(filtered3, cmap='gray')
    axs[1,1].set_title('AVG (9,9)')
    axs[1,1].set_xticklabels([])
    axs[1,1].set_yticklabels([])

    fig.show()
    plt.show()

if __name__=='__main__':
    main()