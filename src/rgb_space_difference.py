import numpy as np
import matplotlib.pyplot as plt
import cv2
from src.utils import display_image

def get_average_image(target_img: np.ndarray, base_img: np.ndarray, alpha = 0.5, show = False) -> np.ndarray:
    """
    Find the weighted average of the image.
    :param target_img: The input image represented as numpy array.
    :param base_img: The image of the empty beach represented as numpy array.
    :param alpha: Weight parameter for calculating how much of the input image will appear in the average.
    :param show: Parameter indicating whether to show the results or not.
    :return: The weighted average of the input images.
    """
    beta = 1 - alpha
    avg_img = cv2.addWeighted(target_img, alpha, base_img, beta, 0.0)
    if show:
        display_image(avg_img)
        plt.title("Average of images")
        plt.show()
    return avg_img

def colorspace_distance_mask(target_img: np.ndarray, base_img: np.ndarray, thresh: int, show = False, morph = False) -> np.ndarray:
    """
    Find the difference between the base and target RGB image in each color channel
    and return the combined results as a single channel mask.
    :param target_img: The averaged input image represented as numpy array.
    :param base_img: The image of the empty beach represented as numpy array.
    :param thresh: Parameter to determine acceptable distances in all the color channels.
    :param show: Parameter indicating whether to show the results or not.
    :param morph: Parameter indicating whether to use morphological processing on the output
    image
    :return: The single channel mask indicating differences in each color channel.
    """
    diff = np.zeros_like(target_img)
    diff_single= np.zeros_like(target_img[:,:,0])
    for i in range(3):
        diff[:,:,i] = cv2.absdiff(target_img[:,:,i], base_img[:,:,i])
        diff[:,:,i][diff[:,:,i] > thresh] = 255
        diff[:,:,i][diff[:,:,i] < thresh] = 0

    diff_single = cv2.bitwise_and(diff[:,:,0], diff[:,:,1])
    diff_single = cv2.bitwise_and(diff[:,:,1], diff[:,:,2])
    if morph:
        # Helps remove noise
        mask_morph = cv2.morphologyEx(diff_single, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        output = cv2.bitwise_and(diff_single, diff_single, None, mask_morph)
        if show:
            display_image(output)
            plt.title("Colorspace difference mask")
            plt.show()
        return output
    if show:
        display_image(diff_single)
        plt.title("Colorspace difference mask")
        plt.show()
    return diff_single
