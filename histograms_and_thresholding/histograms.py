import cv2 as cv
import matplotlib.pyplot as plt
import os

def main():
    path = os.path.join('..', os.getcwd(), 'images', 'lena.jpg')
    img = cv.imread(path)
    #cv.imshow('Original', img)

    # Histogram of a grayscale image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #cv.imshow('Grayscale', gray)

    gray_hist = cv.calcHist([gray], [0], None, [256], [0, 256])

    fig, axs = plt.subplots(2,3)

    axs[0,0].imshow(gray, cmap='gray')
    axs[0,0].set_title('Image')

    axs[0,1].bar(range(256), gray_hist.flatten(), color='blue')
    axs[0,1].set_title('Grayscale histogram')
    axs[0,1].set_xlabel('Bins')
    axs[0,1].set_ylabel('Nummber of pixels')
    axs[0,1].set_xlim([0,255])

    axs[0,2].set_visible(False)
    
    # Histogram of a color image
    colors = ('blue', 'green', 'red')
    for i,col in enumerate(colors):
        hist = cv.calcHist([img], [i], None, [256], [0,256])
        axs[1,i].bar(range(256), hist.flatten(), color=col)
        axs[1,i].set_title(f'{col} histogram')
        axs[1,i].set_xlabel('Bins')
        axs[1,i].set_ylabel('Nummber of pixels')
        axs[1,i].set_xlim([0,255])

    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    fig.show()

    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    main()