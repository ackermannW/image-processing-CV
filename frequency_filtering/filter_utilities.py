import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def _frequency_mesh(shape):
    M, N = shape
    u = np.arange(M) - M // 2
    v = np.arange(N) - N // 2
    return np.meshgrid(u, v, indexing='ij')


def gaussian_low_pass(shape, D0=30):
    U, V = _frequency_mesh(shape)
    D = np.sqrt(U**2 + V**2)
    return np.exp(-D**2 / (2 * D0**2)).astype(np.float32)


def gaussian_high_pass(shape, D0=30):
    return (1 - gaussian_low_pass(shape, D0)).astype(np.float32)


def gaussian_band_pass(shape, C0=90, W=200):
    U, V = _frequency_mesh(shape)
    D = np.sqrt(U**2 + V**2)
    return np.exp(-((D**2 - C0**2) / (D * W + 0.01))**2).astype(np.float32)


def gaussian_band_reject(shape, C0=90, W=200):
    return (1 - gaussian_band_pass(shape, C0, W)).astype(np.float32)


def butterworth_low_pass(shape, D0=30, order=1):
    U, V = _frequency_mesh(shape)
    D = np.sqrt(U**2 + V**2)
    return (1 / (1 + (D**2 / D0**2))).astype(np.float32)


def butterworth_high_pass(shape, D0=30, order=1):
    return (1 - butterworth_low_pass(shape, D0, order)).astype(np.float32)


def butterworth_band_pass(shape, C0=90, W=200):
    U, V = _frequency_mesh(shape)
    D = np.sqrt(U**2 + V**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        H = 1 - 1 / (1 + D * W / (D**2 - C0**2))**2
    return np.nan_to_num(H, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def butterworth_band_reject(shape, C0=90, W=300):
    U, V = _frequency_mesh(shape)
    D = np.sqrt(U**2 + V**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        H = 1 / (1 + D * W / ((D**2 - C0**2)**2))
    return np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def ideal_low_pass(shape, radius=10):
    M, N = shape
    kernel = np.zeros(shape=shape, dtype=np.uint8)
    cv.circle(kernel, (N // 2, M // 2), radius, 1, -1)
    return kernel.astype(np.float32)


def ideal_high_pass(shape, radius=10):
    return (np.ones(shape=shape, dtype=np.float32) - ideal_low_pass(shape, radius)).astype(np.float32)


def ideal_band_pass(shape, C0=100, W=30):
    U, V = _frequency_mesh(shape)
    D = np.sqrt(U**2 + V**2)
    kernel = np.zeros(shape=shape, dtype=np.float32)
    mask = (D >= C0 - W / 2) & (D <= C0 + W / 2)
    kernel[mask] = 1.0
    return kernel


def ideal_band_reject(shape, C0=100, W=30):
    return (1 - ideal_band_pass(shape, C0, W)).astype(np.float32)


def laplacian_filter(shape):
    U, V = _frequency_mesh(shape)
    return (-4 * np.pi * np.pi * (U**2 + V**2)).astype(np.float32)


def notch_reject_filter(shape, d0=9, u_k=0, v_k=0):
    P, Q = shape
    u = np.arange(P)
    v = np.arange(Q)
    U, V = np.meshgrid(u, v, indexing='ij')
    D_uv = np.sqrt((U - P / 2 + u_k)**2 + (V - Q / 2 + v_k)**2)
    D_muv = np.sqrt((U - P / 2 - u_k)**2 + (V - Q / 2 - v_k)**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        H = (1 / (1 + (d0 / D_uv)**4)) * (1 / (1 + (d0 / D_muv)**4))
    return np.nan_to_num(H, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def plotFilterKernel(kernel):
    shape = kernel.shape
    u, v = np.mgrid[-1:1:2.0/shape[0], -1:1:2.0/shape[1]]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(u, v, kernel, 500, cmap='binary')
    ax.view_init(60, 35)
    return fig


def dft2Image(image):  # Optimal extended fast Fourier transform
    mask = np.ones(image.shape)
    mask[1::2, ::2] = -1
    mask[::2, 1::2] = -1
    fImage = image * mask

    rows, cols = image.shape[:2]
    rPadded = cv.getOptimalDFTSize(rows)
    cPadded = cv.getOptimalDFTSize(cols)

    dftImage = np.zeros((rPadded, cPadded, 2), np.float32)
    dftImage[:rows, :cols, 0] = fImage
    cv.dft(dftImage, dftImage, cv.DFT_COMPLEX_OUTPUT)
    return dftImage

    