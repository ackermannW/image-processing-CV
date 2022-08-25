import numpy as np
import cv2 as cv

def main():
    rectangle = np.zeros((300, 300), dtype="uint8")
    cv.rectangle(rectangle, (25,25), (275,275), 255, -1) # we draw 250x250 white rectangle
    cv.imshow("Rectangle", rectangle)

    circle = np.zeros((300, 300), dtype="uint8")
    cv.circle(circle, (150, 150), 150, 255, -1)
    cv.imshow("Circle", circle)

    bitwise_and = cv.bitwise_and(rectangle, circle)
    cv.imshow("AND", bitwise_and)

    bitwise_or = cv.bitwise_or(rectangle, circle)
    cv.imshow("OR", bitwise_or)

    bitwise_xor = cv.bitwise_xor(rectangle, circle)
    cv.imshow("XOR", bitwise_xor)

    bitwise_not = cv.bitwise_not(circle)
    cv.imshow("NOT", bitwise_not)

    cv.waitKey(0)


if __name__ == '__main__':
    main()
