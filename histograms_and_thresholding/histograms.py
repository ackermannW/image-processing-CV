import cv2 as cv
import matplotlib.pyplot as plt
import os

def main():
    path = os.path.join('..', os.getcwd(), 'images', 'lena.jpg')
    img = cv.imread(path)
    cv.imshow('Original', img)

    # Histogram of a grayscale image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Grayscale', gray)

    gray_hist = cv.calcHist([gray], [0], None, [256], [0, 256])
    plt.figure()
    plt.title('Grayscale histogram')
    plt.xlabel('Bins')
    plt.ylabel('Nummber of pixels')
    plt.plot(gray_hist)
    plt.xlim([0,256])

    plt.show()

    # Histogram of a color image
    colors = ('b', 'g', 'r')
    for i,col in enumerate(colors):
        hist = cv.calcHist([img], [i], None, [256], [0,256])
        plt.plot(hist,color=col)
        plt.xlim([0,256])

    plt.show()

    cv.waitKey(0)

if __name__ == '__main__':
    main()