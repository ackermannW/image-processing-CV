import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import sys

def main():
    script_path = os.path.abspath(sys.argv[0])
    path = os.path.abspath(os.path.join(os.path.dirname(script_path), '..', '..', 'images', 'lena.jpg'))
    imgGray  = cv.imread(path, flags=0) # flags=0 to import as grrayscale
    f = np.fft.fft2(imgGray)
    f_shift = np.fft.fftshift(f)

    M,N = f.shape
    D0 = 30
    u = np.arange(M) - M // 2
    v = np.arange(N) - N // 2
    U, V = np.meshgrid(u, v, indexing='ij')
    D = np.sqrt(U**2 + V**2)
    H = (1 - 1 / (1 + (D**2) / (D0**2))).astype(np.float32)
    
    G_shift = f_shift * H
    G = np.fft.ifftshift(G_shift)
    
    g = np.abs(np.fft.ifft2(G))

    fig = plt.figure(figsize=(16, 8))

    axs = []

    for i in range(1, 7):
        axs.append(fig.add_subplot(2, 4, i))

    axs[0].imshow(imgGray, cmap='gray')
    axs[0].set_title('Original image')

    axs[1].imshow(np.log1p(np.abs(f_shift)), cmap='gray')
    axs[1].set_title('Magnitude spectrum')

    axs[2].imshow(H, cmap='gray')
    axs[2].set_title('Buttherworth high pass')

    axs[3].imshow(np.log1p(np.abs(G_shift)), cmap='gray')
    axs[3].set_title('Centered filtered FT')

    axs[4].imshow(np.log1p(np.abs(G)), cmap='gray')
    axs[4].set_title('Shifted filtered FT')

    axs[5].imshow(g, cmap='gray')
    axs[5].set_title('Restored image')

    # --- 3D subplot (last two columns) ---
    ax3d = fig.add_subplot(2, 4, (7, 8), projection='3d')

    U, V = np.meshgrid(np.arange(M), np.arange(N))
    ax3d.plot_surface(U, V, H, cmap='viridis', edgecolor='none')

    ax3d.set_title('3D Filter')
    ax3d.set_xlabel('U')
    ax3d.set_ylabel('V')
    ax3d.set_zlabel('H(u,v)')

    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    main()
