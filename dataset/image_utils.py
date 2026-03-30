import numpy as np
import cv2

def create_gradient_image(size=(128, 128), direction='horizontal'):
    """
    Create a simple gradient image (horizontal or vertical).
    """
    if direction == 'horizontal':
        gradient = np.tile(np.linspace(0, 255, size[1], dtype=np.uint8), (size[0], 1))
    else:
        gradient = np.tile(np.linspace(0, 255, size[0], dtype=np.uint8), (size[1], 1)).T
    img = np.stack([gradient]*3, axis=2)
    return img

def create_pattern_image(size=(128, 128)):
    """
    Create a simple checkerboard pattern image.
    """
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    s = 16
    for i in range(size[0]):
        for j in range(size[1]):
            if ((i//s) % 2) == ((j//s) % 2):
                img[i, j] = [255, 255, 255]
            else:
                img[i, j] = [0, 0, 0]
    return img
