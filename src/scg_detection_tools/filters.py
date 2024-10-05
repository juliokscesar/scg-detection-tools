from abc import ABC, abstractmethod
from typing import Tuple
import logging
import numpy as np
import supervision as sv

def filter_detections_xyxy(detections: sv.Detections, intersection_thresh: float = 0.9, imghw=(640,640)):
    """ 
    Filter possible duplicates in detection boxes by calculating their area of intersection and comparing that over the area of the smallest one of the boxes.
    If intersection >= min(Area) * intersection_thresh, the smallest box is considered to be a duplicate and gets removed.
    """
    remove = []
    boxes = detections.xyxy.astype(np.int32)
    areas = detections.box_area.astype(np.int32)

    grid = np.full(shape=imghw, fill_value=-1)

    for i in range(len(boxes)):
        # Skip index if already to be removed
        if i in remove:
            continue

        # Find it's place on the grid and check if there is another box on it
        x1, y1, x2, y2 = boxes[i]
        grid_place = grid[y1:(y2+1), x1:(x2+1)]
        intersect_box_idx = None
        for row in grid_place:
            for boxidx in row:
                # -1 indicates that no other box was in there
                if boxidx != -1:
                    intersect_box_idx = boxidx
                    break
            if intersect_box_idx is not None:
                break
        if intersect_box_idx is not None:
            # get intersection box
            other_x1, other_y1, other_x2, other_y2 = boxes[intersect_box_idx]
            inter_x1 = max(x1, other_x1)
            inter_y1 = max(y1, other_y1)
            inter_x2 = min(x2, other_x2)
            inter_y2 = min(y2, other_y2)
            # skip if no intersection

            # Get smallest area of the boxes and compare with the intersection area
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            small_idx, small_area = sorted([(i,areas[i]), (intersect_box_idx,areas[intersect_box_idx])], key=lambda t: t[1])[0]
            if inter_area >= (small_area * intersection_thresh):
                remove.append(small_idx)
            # if the 'i' box wasnt removed, we need to update the grid place
            if small_idx != i:
                grid[y1:(y2+1), x1:(x2+1)] = i
        else:
            # Put the box index in its correspondent grid place
            grid[y1:(y2+1), x1:(x2+1)] = i
                                  

    # for i in range(len(boxes)):
    #     # Skip index if already to be removed
    #     if i in remove:
    #         continue

    #     for j in range(i+1, len(boxes)):
    #         # Skip index if already to be removed
    #         if j in remove:
    #             continue

    #         # Find a intersection box
    #         x_a1, y_a1, x_a2, y_a2 = boxes[i]
    #         x_b1, y_b1, x_b2, y_b2 = boxes[j]
    #         inter_x1 = max(x_a1, x_b1)
    #         inter_y1 = max(y_a1, y_b1)
    #         inter_x2 = min(x_a2, x_b2)
    #         inter_y2 = min(y_a2, y_b2)
    #         # Skip if no intersection
    #         if not ((inter_x1 < inter_x2) and (inter_y1 < inter_y2)):
    #             continue
    #         inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    #         # Take smallest box
    #         small_idx = small_area = 1.9e+9
    #         for box_idx in [i, j]:
    #             box_area = (boxes[box_idx][2] - boxes[box_idx][0]) * (boxes[box_idx][3] - boxes[box_idx][1])
    #             if box_area < small_area:
    #                 small_area = box_area
    #                 small_idx = box_idx

    #         # If intersection is bigger or equal to the given percentage of the smallest area, remove box
    #         if inter_area >= (small_area * intersection_thresh):
    #             remove.append(small_idx)
    if len(remove) == 0:
        return detections
    # Sort in reverse order so it is safe to delete them all from the boxes
    remove = sorted(remove, reverse=True)
    box_classes = detections.class_id
    box_confidence = detections.confidence
    for idx in remove:
        if idx >= len(boxes):
            logging.warning(f"In filter_detections_xyxy: index {idx} in remove list, but boxes length is {len(boxes)}")
            continue
        boxes = np.delete(boxes, idx, axis=0)
        box_classes = np.delete(box_classes, idx)
        box_confidence = np.delete(box_confidence, idx)
    new_det = sv.Detections(xyxy=boxes, class_id=box_classes, confidence=box_confidence)
    return new_det

class DetectionsFilter(ABC):
    @abstractmethod
    def filter(self, detections: sv.Detections) -> sv.Detections:
        pass

    def __call__(self, detections: sv.Detections) -> sv.Detections:
        return self.filter(detections)

class DetectionFilterDuplicates(DetectionsFilter):
    def __init__(self, intersection_tresh: float = 0.95, imghw=(640,640)):
        self._intersection_thresh = intersection_tresh
        self._imghw = imghw

    def filter(self, detections: sv.Detections) -> sv.Detections:
        """ 
        Filter possible duplicates in detection boxes by calculating their area of intersection and comparing that over the area of the smallest one of the boxes.
        If intersection >= min(Area) * intersection_thresh, the smallest box is considered to be a duplicate and gets removed.
        """
        remove = []
        boxes = detections.xyxy.astype(np.int32)
        areas = detections.box_area.astype(np.int32)

        grid = np.full(shape=self._imghw, fill_value=-1)

        for i in range(len(boxes)):
            # Skip index if already to be removed
            if i in remove:
                continue

            # Find it's place on the grid and check if there is another box on it
            x1, y1, x2, y2 = boxes[i]
            grid_place = grid[y1:(y2+1), x1:(x2+1)]
            intersect_box_idx = None
            for row in grid_place:
                for boxidx in row:
                    # -1 indicates that no other box was in there
                    if boxidx != -1:
                        intersect_box_idx = boxidx
                        break
                if intersect_box_idx is not None:
                    break
            if intersect_box_idx is not None:
                # get intersection box
                other_x1, other_y1, other_x2, other_y2 = boxes[intersect_box_idx]
                inter_x1 = max(x1, other_x1)
                inter_y1 = max(y1, other_y1)
                inter_x2 = min(x2, other_x2)
                inter_y2 = min(y2, other_y2)
                # skip if no intersection

                # Get smallest area of the boxes and compare with the intersection area
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                small_idx, small_area = sorted([(i,areas[i]), (intersect_box_idx,areas[intersect_box_idx])], key=lambda t: t[1])[0]
                if inter_area >= (small_area * self._intersection_thresh):
                    remove.append(small_idx)
                    # if the 'i' box wasnt removed, we need to update the grid place
                    if small_idx != i:
                        grid[y1:(y2+1), x1:(x2+1)] = i
                else: # if no box is removed, we need to update the grid with 'i' on its correspondent place, skipping where intersect_box_idx is already on
                    for y in range(y1, y2):
                        for x in range(x1, x2):
                            if grid[y,x] == intersect_box_idx:
                                continue
                            grid[y,x] = i
            else:
                # Put the box index in its correspondent grid place
                grid[y1:(y2+1), x1:(x2+1)] = i
                                    
        if len(remove) == 0:
            return detections
        box_classes = detections.class_id
        box_confidence = detections.confidence

        remove_mask = np.ones(len(boxes), dtype=bool)
        remove_mask[remove] = False
        boxes = boxes[remove_mask]
        box_classes = box_classes[remove_mask]
        box_confidence = box_confidence[remove_mask]

        new_det = sv.Detections(xyxy=boxes, class_id=box_classes, confidence=box_confidence)
        return new_det
        

class DetectionFilterSize(DetectionsFilter):
    def __init__(self, max_obj_wh: Tuple[int,int]):
        self._max_obj_wh = max_obj_wh
    
    def filter(self, detections: sv.Detections) -> sv.Detections:
        remove = []
        boxes = detections.xyxy.astype(np.int32)
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            if ((x2-x1) > self._max_obj_wh[0]) or ((y2-y1) > self._max_obj_wh[1]):
                remove.append(i)

        if len(remove) == 0:
            return detections
        box_classes = detections.class_id
        box_confidence = detections.confidence
        
        remove_mask = np.ones(len(boxes), dtype=bool)
        remove_mask[remove] = False
        boxes = boxes[remove_mask]
        box_classes = box_classes[remove_mask]
        box_confidence = box_confidence[remove_mask]

        new_det = sv.Detections(xyxy=boxes, class_id=box_classes, confidence=box_confidence)
        return new_det