from abc import ABC, abstractmethod
from typing import Tuple
import logging
import numpy as np
import supervision as sv

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
            # clamp their values to avoid any index out of bounds
            x1, y1, x2, y2 = boxes[i]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(x2, grid.shape[1])
            y2 = min(y2, grid.shape[0])

            grid_place = grid[y1:(y2+1), x1:(x2+1)]
            intersect_box_idx = []
            for row in grid_place:
                for boxidx in row:
                    # -1 indicates that no other box was in there
                    if (boxidx != -1) and (boxidx not in intersect_box_idx):
                        intersect_box_idx.append(boxidx)
            if len(intersect_box_idx) != 0:
                for int_idx in intersect_box_idx:
                    # get intersection box
                    other_x1, other_y1, other_x2, other_y2 = boxes[int_idx]
                    inter_x1 = max(x1, other_x1)
                    inter_y1 = max(y1, other_y1)
                    inter_x2 = min(x2, other_x2)
                    inter_y2 = min(y2, other_y2)

                    # Get smallest area of the boxes and compare with the intersection area
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    small_idx, small_area = sorted([(i,areas[i]), (int_idx,areas[int_idx])], key=lambda t: t[1])[0]
                    if inter_area >= (small_area * self._intersection_thresh):
                        remove.append(small_idx)
                        # update the grid place if the 'i' box wasnt removed
                        if small_idx != i:
                            grid[inter_y1:(inter_y2+1),inter_x1:(inter_x2+1)] = i

                    else: # if no box is removed, we need to update the grid with 'i' on its correspondent place, skipping where intersect_box_idx is already on
                        for y in range(y1, y2):
                            for x in range(x1, x2):
                                if grid[y,x] == int_idx:
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