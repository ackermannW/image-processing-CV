import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def butterworthHighPass(width, height, d0):
    xc = width//2
    yc = height//2

    z = np.zeros((width,height), np.float32)
    for i in range(-width//2,width//2-1,1):
        for j in range(-height//2, height//2-1,1):
            z[xc+i,yc+j] =1 - 1/(1+(i**2 + j**2)/d0**2)
    return z    
    

img = cv.imread('frequency_filtering\m1a2_abrams_l5.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

(width,height) = img.shape[:2]
x = np.linspace(-width//2, width//2-1, 1000)
y = np.linspace(-height//2, height//2-1, 1000)
d0 = 50
filter = butterworthHighPass(width, height, d0)

# plot filter in 3d
X,Y = np.meshgrid(x,y)
Z = 1/(1+(X**2 + Y**2)/d0**2)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='gray')
#ax.plot_surface(X,Y,Z, rstride = 1, cstride = 1, cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');
ax.view_init(60, 35)
fig


dft = cv.dft(np.float32(img), cv.DFT_COMPLEX_OUTPUT) #find dft of an image
dftshift = np.fft.fftshift(dft) # center the FT
filteredTransform = dftshift*filter # convolute FT of an image and a filter
f_ishift = np.fft.ifftshift(filteredTransform) # move back to the origin
filteredImage = cv.idft(f_ishift) # inverse FT

fig1 = plt.figure()
plt.subplot(121)
plt.imshow(np.log(np.abs(dftshift)), cmap='gray')
plt.title('FT Original')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(filter, cmap='gray')
plt.title('Butterworth HP filter')
plt.xticks([])
plt.yticks([])

# plotting
fig2 = plt.figure()
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(filteredImage, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Butterworth filtered')
plt.show()
cv.waitKey(0)
