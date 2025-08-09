"""
Code based off
https://surma.dev/things/ditherpunk/
"""
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import cv2
import math

_BAYER_FILTER_0 = np.array([[0, 2], [3, 1]], dtype=np.float64)


def _recursive_bayer_dithering(level: int = 0) -> np.ndarray:
    if level == 0:
        return _BAYER_FILTER_0/4
    next_level = _recursive_bayer_dithering(level-1)*4
    return np.concatenate(
        (
            np.concatenate(
                (next_level*4+_BAYER_FILTER_0[0, 0], next_level*4+_BAYER_FILTER_0[0, 1]), axis=1),
            np.concatenate(
                (next_level*4+_BAYER_FILTER_0[1, 0], next_level*4+_BAYER_FILTER_0[1, 1]), axis=1)
        ),
    )/(4**(level+1))


def bayer_dither(image: np.ndarray, level: int = 0) -> np.ndarray:
    dithering_filter = _recursive_bayer_dithering(level)*255
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)

    repeats = np.array(image.shape)//np.array(np.shape(dithering_filter)) + 1
    tiled_dither_filter = np.tile(
        dithering_filter,
        tuple(repeats)
    )[tuple(map(slice, image.shape))]
    dithered_image = np.where(
        image >= 255-tiled_dither_filter, 255, 0).astype(np.uint8)
    return dithered_image


def _get_tiled_blue_noise(shape: tuple) -> np.ndarray:
    # TODO
    raise NotImplementedError()


def blue_noise_dithering(image: np.ndarray):
    # TODO
    raise NotImplementedError()


def simple_dithering(image, threshold=128):
    # Ensure the image is in grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Apply dithering
    image_size = image.shape[:2]
    noise = (np.random.rand(*image_size)-0.5)*255
    image += noise
    dithered_image = np.where(image >= threshold, 255, 0).astype(np.uint8)

    return dithered_image


def display_images(image):
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # Load an example image
    # image = cv2.imread('data/test_image2.png')
    image = cv2.imread('data/test_image1.jpg')

    # Apply simple dithering
    # dithered = simple_dithering(image)
    if image is not None:
        dithered = bayer_dither(image, level=4)
        # Display the original and dithered images
        display_images(dithered)
    else:
        print("Error: Image could not be loaded. Please check the file path.")
