import cv2 as cv
import os
import sys
import matplotlib.pyplot as plt

def main():
    script_path = os.path.abspath(sys.argv[0])
    path = os.path.abspath(os.path.join(os.path.dirname(script_path), '..', 'images', 'lena.jpg'))
    img = cv.imread(path, flags=0)

    gaussian1 = cv.getGaussianKernel(7,0)
    gaussian2 = cv.getGaussianKernel(9,0)
    gaussian3 = cv.getGaussianKernel(19,0)

    gaussian1 = gaussian1*gaussian1.T
    gaussian2 = gaussian2*gaussian2.T
    gaussian3 = gaussian3*gaussian3.T

    filtered1 = cv.filter2D(src=img, ddepth=-1, kernel=gaussian1)
    filtered2 = cv.filter2D(src=img, ddepth=-1, kernel=gaussian2)
    filtered3 = cv.filter2D(src=img, ddepth=-1, kernel=gaussian3)

    # also possible to do:
    # filtered = cv.GaussianBlur(img, (9,9), 0)

    fig, axs = plt.subplots(3,3)

    axs[0,0].imshow(img, cmap='gray')
    axs[0,0].set_title('Original image')
    axs[0,0].set_xticklabels([])
    axs[0,0].set_yticklabels([])

    axs[0,1].imshow(gaussian1, cmap='gray')
    axs[0,1].set_title('Kernel (7,7)')
    axs[0,1].set_xticklabels([])
    axs[0,1].set_yticklabels([])

    axs[0,2].imshow(filtered1, cmap='gray')
    axs[0,2].set_title('Filtered image')
    axs[0,2].set_xticklabels([])
    axs[0,2].set_yticklabels([])

    axs[1,0].imshow(img, cmap='gray')
    axs[1,0].set_title('Original image')
    axs[1,0].set_xticklabels([])
    axs[1,0].set_yticklabels([])

    axs[1,1].imshow(gaussian2, cmap='gray')
    axs[1,1].set_title('Kernel (9,9)')
    axs[1,1].set_xticklabels([])
    axs[1,1].set_yticklabels([])

    axs[1,2].imshow(filtered2, cmap='gray')
    axs[1,2].set_title('Filtered image')
    axs[1,2].set_xticklabels([])
    axs[1,2].set_yticklabels([])

    axs[2,0].imshow(img, cmap='gray')
    axs[2,0].set_title('Original image')
    axs[2,0].set_xticklabels([])
    axs[2,0].set_yticklabels([])

    axs[2,1].imshow(gaussian3, cmap='gray')
    axs[2,1].set_title('Kernel (19,19)')
    axs[2,1].set_xticklabels([])
    axs[2,1].set_yticklabels([])

    axs[2,2].imshow(filtered3, cmap='gray')
    axs[2,2].set_title('Filtered image')
    axs[2,2].set_xticklabels([])
    axs[2,2].set_yticklabels([])

    fig.show()
    plt.show()

if __name__=='__main__':
    main()