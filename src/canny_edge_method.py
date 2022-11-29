import numpy as np
import matplotlib.pyplot as plt
import cv2
from src.utils import display_image, binarize_by_otsu

def get_canny_edges(target_img: np.ndarray, base_img: np.ndarray, otsu = False, normalize = False, show = False) -> np.ndarray:
    """
	Find the Canny edges from difference between the base empty beach image and
    target and return the combined results as a single channel mask.
	:param target_img: The input image represented as numpy array.
    :param base_img: The image of the empty beach represented as numpy array.
    :param otsu: Parameter to determine whether to apply otsu binarization before edge detetction.
    :param normalize: Parameter to determine whether to normalize before edge detetction.
	:param show: Parameter indicating whether to show the results or not.
    :param morph: Parameter indicating whether to use morphological processing on the output
    image
	:return: The single channel mask including edges from the image differences.
	"""
    x = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY) # x is the target image
    y = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)   # y is the base image

    # Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    x = clahe.apply(x)

    # Filter the target to smooth sunny artifacts 
    x = cv2.bilateralFilter(x,9,127,127)

    x = cv2.absdiff(x, y)

    if normalize:
        x = ((x - np.min(x) / np.max(x) - np.min(x)) * 255.0).astype('uint8')

    if otsu:
        x = binarize_by_otsu(x)

    x= cv2.Canny(x, 100, 200)
    x = cv2.dilate(x, np.ones((3,3)))
    x = cv2.morphologyEx(x, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1)))
    if show:
        display_image(x)
        plt.title("Canny edges of the difference image")
        plt.show()
    return x