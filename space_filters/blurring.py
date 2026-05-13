import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------
# Noise generation functions
# ------------------------------------------------------------

def add_gaussian_noise(image, mean=0, sigma=20):
    noise = np.random.normal(mean, sigma, image.shape)

    noisy = image.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255)

    return noisy.astype(np.uint8)


def add_salt_pepper_noise(image, amount=0.05):
    noisy = image.copy()

    # Number of pixels to modify
    num_pixels = int(amount * image.size)

    # Salt noise
    coords = (
        np.random.randint(0, image.shape[0], num_pixels),
        np.random.randint(0, image.shape[1], num_pixels)
    )
    noisy[coords] = 255

    # Pepper noise
    coords = (
        np.random.randint(0, image.shape[0], num_pixels),
        np.random.randint(0, image.shape[1], num_pixels)
    )
    noisy[coords] = 0

    return noisy

def show_images(images, titles, cols=3, figsize=(14, 8)):
    rows = int(np.ceil(len(images) / cols))

    plt.figure(figsize=figsize)

    for i, (img, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(rows, cols, i)

        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def main():

    image_path = (
        Path(__file__).resolve().parent.parent
        / "images"
        / "lena.jpg"
    )

    img = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    # --------------------------------------------------------
    # Add noise
    # --------------------------------------------------------

    gaussian_noisy = add_gaussian_noise(img)

    salt_pepper_noisy = add_salt_pepper_noise(img)

    # --------------------------------------------------------
    # Filtering
    # --------------------------------------------------------

    # Mean filter
    average_filtered = cv.blur(gaussian_noisy, (5, 5))

    # Gaussian filter
    gaussian_filtered = cv.GaussianBlur(
        gaussian_noisy,
        (5, 5),
        sigmaX=1.0
    )

    # Median filter (excellent for salt & pepper noise)
    median_filtered = cv.medianBlur(salt_pepper_noisy, 5)

    images = [
        img,
        gaussian_noisy,
        average_filtered,
        gaussian_filtered,
        salt_pepper_noisy,
        median_filtered
    ]

    titles = [
        "Original image",
        "Gaussian noise",
        "Average filter",
        "Gaussian filter",
        "Salt & Pepper noise",
        "Median filter"
    ]

    show_images(images, titles)


if __name__ == "__main__":
    main()