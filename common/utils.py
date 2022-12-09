import cv2
import os
import numpy as np

def open_rgby(path, id):
    colors = ['red', 'green', 'blue', 'yellow']
    flags = cv2.IMREAD_GRAYSCALE
    img = [cv2.imread(os.path.join(path, id + '_' + color + '.png'), flags).astype(
                        np.float32) / 255 for color in colors]
    return np.stack(img, axis=-1)
