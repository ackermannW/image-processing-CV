import cv2 as cv
import matplotlib.pyplot as plt
import os
import sys

def main():
    script_path = os.path.abspath(sys.argv[0])
    path = os.path.abspath(os.path.join(os.path.dirname(script_path), '..', 'images', 'lena.jpg'))
    img = cv.imread(path)

    # Display original image
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')

    # Convert to grayscale and display
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    plt.subplot(2, 3, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale')
    plt.axis('off')

    # Calculate histogram and display
    plt.subplot(2, 3, 3)
    plt.hist(gray.ravel(), bins=256, range=[0, 256], color='r')
    plt.title('Grayscale Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    # Apply thresholding and display
    ret, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
    plt.subplot(2, 3, 4)
    plt.imshow(thresh, cmap='gray')
    plt.title('Thresholded')
    plt.axis('off')

    # Apply inverse thresholding and display
    ret, thresh_inv = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)
    plt.subplot(2, 3, 5)
    plt.imshow(thresh_inv, cmap='gray')
    plt.title('Thresholded Inverse')
    plt.axis('off')

    # Apply adaptive thresholding and display
    adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)
    plt.subplot(2, 3, 6)
    plt.imshow(adaptive_thresh, cmap='gray')
    plt.title('Adaptive Thresholding')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    print("Press 0 to close.")
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
