import numpy as np
import matplotlib.pyplot as plt
import cv2

def display_image(input_img: np.ndarray):
    """
    Utility function to display the image (BGR) from OpenCV
    in the proper way.
    :param input_img: The input image represented as numpy array in BGR or GRAYSCALE colors.
    """
    if len(input_img.shape) == 3:
        plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(input_img, cmap='gray')
    plt.axis('off')

def binarize_by_thresholding(img: np.ndarray, threshold: float) -> np.ndarray:
    """
    Returns a binary version of the image by applying a thresholding operation.
    :param img: The input image.
    :param threshold: Threshold to determine at what point binarization is done.
    :return: binarized image based on the threshold.
    """
    return ((img >= threshold)*255).astype('uint8')

def binarize_by_otsu(img: np.ndarray) -> np.ndarray:
    """
    Returns a binary version of the image by applying a thresholding operation.
    :param img: The input image.
    :return: binarized image based on the found optimum threshold.
    """
    otsu_threshold = 0
    lowest_criteria = np.inf
    for threshold in range(255):
        thresholded_im = img >= threshold
        # compute weights
        weight1 = np.sum(thresholded_im) / img.size
        weight0 = 1 - weight1

        # if one the classes is empty, that threshold will not be considered
        if weight1 != 0 and weight0 != 0:
            # compute criteria, based on variance of these classes
            var0 = np.var(img[thresholded_im == 0])
            var1 = np.var(img[thresholded_im == 1])
            otsu_criteria = weight0 * var0 + weight1 * var1

            if otsu_criteria < lowest_criteria:
                otsu_threshold = threshold
                lowest_criteria = otsu_criteria

    return binarize_by_thresholding(img, otsu_threshold)

def get_bboxes_and_points(img:np.ndarray, mask:np.ndarray, cutoff = 450, minBox = None, maxBox = None, minRatio = None, 
                                        maxRatio = None, draw_contours = False, draw_boxes = False, return_image = False):
    """
    Finds the bounding boxes and the central point by finding contours from
    the input mask. Optionally, draws the contours or boxes on the input image
    that corresponds to the mask.
    :param img: Input image corresponding to the mask.
    :param mask: The mask for finding contours.
    :param cutoff: Cutoff height of the image that is not considered.
    :param minBox: Minimum acceptable area for the bounding box.
    :param maxBox: Maximum acceptable area for the bounding box.
    :param minRatio: Minimum aspect ratio of the boxes.
    :param maxRatio: Maximum aspect ratio of the boxes.
    :param draw_contours: Determines whether to draw and show the detected contours.
    :param draw_boxes: Determines whether to draw and show the detected boxes.
    :param return_image: Determines whether to return the drawn over image.
    :return: 
        List of the detected boxes
        List of the detected points
        Drawn over image or 0
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    show_img = img.copy()
    points = []
    boxes = []
    for cnt in contours:
        M = cv2.moments(cnt)
        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        except:
            continue
        x,y,w,h=cv2.boundingRect(cnt)
        if y > cutoff and y+h > cutoff and h>0 and w>0:
            if minBox and h*w < minBox:
                continue
            if maxBox and h*w > maxBox:
                continue
            if minRatio and w/h < minRatio:
                continue
            if maxRatio and w/h > maxRatio:
                continue
            boxes.append([x,y,w,h])
            points.append((cX, cY))
            if draw_boxes:
                cv2.rectangle(show_img, (x,y), (x+w,y+h), (255,0,0), 2)
            if draw_contours:
                cv2.drawContours(show_img, [cnt], 0, (0,0,255), 3)
    if draw_boxes or draw_contours:
        display_image(show_img)
        plt.title("Detected regions")
        plt.show()
    if return_image and draw_boxes or draw_contours:
        return boxes, points, show_img
    return boxes, points, 0

# credits: https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def bb_intersection_over_union(boxA, boxB):
    """Calculates the IoU for the input boxes"""
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou
