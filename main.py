import cv2
import numpy as np
from matplotlib import pyplot as plt

def analyze_frequency(image_path):
    img = cv2.imread(image_path, 0) # Carrega em escala de cinza
    dft = np.fft.fft2(img)          # Transformada de Fourier
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift))
    return magnitude_spectrum