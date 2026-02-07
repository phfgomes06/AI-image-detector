import cv2
import numpy as np
from matplotlib import pyplot as plt

def get_spectrum_image(image_path):
    img = cv2.imread(image_path, 0)
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift + 1))
    
    spectrum_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    spectrum_final = np.uint8(spectrum_normalized)

    return spectrum_final

