import numpy as np
from typing import Tuple

def cvt_segment_to_box(seg_contour: np.ndarray):
    pass


# Convert every contour in contours to a mask image of size imgsz
# The mask image is a black and white image where every contour region
# is white
# Each contour in 'contours' contains N points that describes the contour
# but FLATTENED: countour = x0, y0, x1, y1, ..., xn, yn
def cvt_contours_to_mask(contours: np.ndarray, imgsz: Tuple[int], normalized=True):
    maskimg = np.zeros(imgsz, dtype=np.uint8)
    
    for contour in contours:
        # reshape list into pairs of (x,y) coordinates
        points = contour.reshape(-1, 2)
        if normalized:
            points = np.int32(points * np.array(imgsz[::-1]))
        else:
            points = points.astype(np.int32)

        cv2.fillPoly(maskimg, [points], color=255)

    return maskimg

