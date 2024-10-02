import numpy as np
from typing import Tuple
import cv2
import numbers

def segment_to_box(seg_contour: np.ndarray, normalized=False, imgsz: Tuple[int,int]=None):
    if normalized and imgsz is None:
        raise ValueError("For normalized contours, imgsz argument is required")
    # reshape into array of (Npoinst, 2) from (x1 y1 x2 y2 x3 y3 ....)
    if not isinstance(seg_contour, np.ndarray):
        seg_contour = np.array(seg_contour)
    if seg_contour.ndim != 2:
        seg_contour = seg_contour.reshape(len(seg_contour) // 2, 2)

    x1, y1 = np.min(seg_contour, axis=0)
    x2, y2 = np.max(seg_contour, axis=0)
    
    h, w = imgsz
    if normalized:
        x1 *= w
        x2 *= w
        y1 *= h
        y2 *= h

    return np.array([x1, y1, x2, y2]).astype(np.int32)


# Convert every contour in contours to masks images of size imgsz
# The mask image is a black and white image where every contour region
# is white
# Each contour in 'contours' contains N points that describes the contour
# but FLATTENED: countour = x0, y0, x1, y1, ..., xn, yn
def contours_to_masks(contours: list, imgsz: Tuple[int, int], normalized=True, binary_mask=True):
    masks = []

    for contour in contours:
        # if contour comes as a flat array reshape list into pairs of (x,y) coordinates
        if isinstance(contour[0], numbers.Number):
            points = np.array(contour).reshape(len(contour)//2, 2)
        else:
            points = np.array(contour.copy())

        if normalized:
            points = np.int32(points * np.array(imgsz[::-1]))
        else:
            points = points.astype(np.int32)

        maskimg = np.zeros(imgsz, dtype=np.uint8)
        cv2.fillPoly(maskimg, [points], color=255)
        if binary_mask:
            maskimg = np.where(maskimg == 255, 1, 0)
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


def detbox_to_yolo_fmt(boxes: np.ndarray, imgsz: Tuple[int], normalized=False):
    yolo_boxes = []
    imgw, imgh = imgsz
    for box in boxes:
        x1, y1, x2, y2 = box
        # yolo dataset format is: class x_center y_center width height (all normalized)
        x_center = float((x2 - x1)/2 + x1)
        y_center = float((y2 - y1)/2 + y1)
        width = float((x2-x1))
        height = float((y2-y1))
        if not normalized:
            x_center /= imgw
            y_center /= imgh
            width /= imgw
            height /= imgh
        yolo_boxes.append([x_center, y_center, width, height])

    return yolo_boxes

def yolo_boxes_to_absolute(yolo_fmt_boxes: np.ndarray, imgsz: Tuple[int]) -> np.ndarray:
    abs_boxes = []
    imgw, imgh = [float(x) for x in imgsz]
    for box in yolo_fmt_boxes:
        xcenter, ycenter, width, height = box
        x1 = xcenter - width
        y1 = ycenter - height
        x2 = xcenter + width
        y2 = ycenter + height
        abs_boxes.append(np.array([x1 * imgw, y1 * imgh, x2 * imgw, y2 * imgh]))
    return np.array(abs_boxes).astype(np.int32)
        

def contour_to_yolo_fmt(contours: list, imgsz: Tuple[int], normalized=False):
    """ 
    Returns contours in YOLO format (normalized x1 x2 ... xn)
    Contours must be a list of every contour in the image.
    Contour is a list of points (x, y) of that contour
    """
    yolo_contours = []
    imgw, imgh = imgsz
    
    for contour in contours:
        if not isinstance(contour, np.ndarray):
            contour = np.array(contour)
        contour = contour.astype(np.float64)
        if not normalized:
            for point in contour:
                point[0] /= float(imgw)
                point[1] /= float(imgh)
        yolo_contours.append(contour.reshape(contour.size))
    return yolo_contours

