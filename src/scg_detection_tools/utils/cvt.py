import numpy as np
from typing import Tuple
import cv2

def segment_to_box(seg_contour: np.ndarray):
    pass


# Convert every contour in contours to masks images of size imgsz
# The mask image is a black and white image where every contour region
# is white
# Each contour in 'contours' contains N points that describes the contour
# but FLATTENED: countour = x0, y0, x1, y1, ..., xn, yn
def contours_to_masks(contours: list, imgsz: Tuple[int], normalized=True):
    masks = []

    for contour in contours:
        # reshape list into pairs of (x,y) coordinates
        points = np.array(contour).reshape(len(contour)//2, 2)

        if normalized:
            points = np.int32(points * np.array(imgsz[::-1]))
        else:
            points = points.astype(np.int32)

        maskimg = np.zeros(imgsz, dtype=np.uint8)
        cv2.fillPoly(maskimg, [points], color=255)
        masks.append(maskimg)

    return masks


def boxes_to_masks(boxes: np.ndarray, imgsz: Tuple[int], normalized=False):
    masks = []
    for box in boxes:
        x1, y1, x2, y2 = box
        points = np.array([[x1,y1], [x2,y1], [x2, y2], [x1, y2]])
        if normalized:
            points = np.int32(points * np.array(imgsz[::-1]))

        maskimg = np.zeros(imgsz, dtype=np.uint8)
        cv2.fillPoly(maskimg, [points], color=255)
        masks.append(maskimg)

    return masks

