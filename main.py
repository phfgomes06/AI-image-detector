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

def get_radial_profile(magnitude_spectrum):
    rows,cols = magnitude_spectrum.shape
    cent_x,cent_y = rows // 2, cols // 2

    y, x = np.indices((rows, cols))

    r = np.sqrt((x - cent_x)**2 + (y - cent_y)** 2)
    r = r.astype(np.int32)

    max_radius = min(cent_x, cent_y)
    tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
    nr = np.bincount(r.ravel())

    radial_profile = tbin[:max_radius] / nr[:max_radius]
    
    return radial_profile

def gradient_analysis(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    img_blur = cv2.GaussianBlur(img, (3,3), 0)
    sobel_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(sobel_x, sobel_y)

    magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    magnitude_final = np.uint8(magnitude_normalized)

    neon_map = cv2.applyColorMap(magnitude_final, cv2.COLORMAP_JET)

    return magnitude_final, neon_map
