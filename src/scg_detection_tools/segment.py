from abc import ABC, abstractmethod
import cv2
import numpy as np
import torch
import sam2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import supervision as sv
from typing import Union, Tuple
import threading
import queue
import matplotlib.pyplot as plt

from scg_detection_tools.detect import BaseDetectionModel, Detector


class BaseSegmentationModel(ABC):
    def __init__(self, model_type: str, underlying_model):
        self._model_type = model_type
        self._underlying_model = underlying_model
    
    @abstractmethod
    def segment(self):
        pass


class SAM2Segment:
    def __init__(self, 
                 sam2_ckpt_path: str, 
                 sam2_cfg: str, 
                 custom_state_dict: str = None,
                 detection_assist_model: BaseDetectionModel = None):
        self._predictor = self._load_sam2(sam2_ckpt_path, sam2_cfg, custom_state_dict)

        self._detection_model = detection_assist_model
        self._detector = None
        if self._detection_model:
            self._detector = Detector(detection_assist_model)

    def _alt_thread_detect(self, img_path: str, use_slice: bool):
        def _call_detection(img_path, use_slice, que):
            detections = self._detector.detect_objects(img_path, use_slice=use_slice)[0]
            que.put(detections)

        result_queue = queue.Queue()
        t = threading.Thread(target=_call_detection, args=(img_path, use_slice, result_queue))
        t.start()
        t.join()
        return result_queue.get()
        
    
    def detect_segment(self,
                       img_path: str,
                       use_slice_detection = False):
        if not self._detector or not self._detection_model:
            raise RuntimeError("SAM2Segment detection assist model needs to be defined to use detect_segment")
        
        detections = self._alt_thread_detect(img_path, use_slice_detection)
        assert(detections is not None)

        print(f"DEBUG segment.py: detected {len(detections.xyxy)} in image {img_path}")
        masks = self.segment_detection(img_path, detections)

        return masks, self._sam2masks_to_contours(masks)


    def slice_segment_detect(self, img_path: str, slice_wh: Tuple[int,int], embed_slice_callback=None):
        result = {
            "original_image": img_path,
            "detections": None,
            "slices": [],
        }
        
        def segment_slice_callback(img_path: str, slice: np.ndarray, slice_path: str, slice_boxes: np.ndarray):
            slice_buffer = {
                "path": slice_path,
                "masks": [],
                "contours": []
            }
            if len(slice_boxes) == 0:
                return

            with torch.no_grad():
                self._predictor.set_image(slice)
                masks, _, _ = self._predictor.predict(point_coords=None,
                                                    point_labels=None,
                                                    box=slice_boxes[None, :],
                                                    multimask_output=False)
                torch.cuda.empty_cache()
        
            contours = self._sam2masks_to_contours(masks)
            slice_buffer["masks"] = masks
            slice_buffer["contours"] = contours

            if embed_slice_callback is not None:
                embed_slice_callback(img_path, slice, slice_path, slice_boxes, masks, contours)

            result["slices"].append(slice_buffer)

        detections = self._detector.detect_objects(img=img_path,
                                                   slice_detect=True,
                                                   embed_slice_callback=segment_slice_callback,
                                                   slice_wh=slice_wh)[0]
        result["detections"] = detections
        return result


    def segment_point(self, img_p: Union[str, np.ndarray], input_points: np.ndarray, input_labels: np.ndarray):
        if isinstance(img_p, str):
            img = cv2.imread(img_p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = img_p

        with torch.no_grad():
            self._predictor.set_image(img)
            masks, _, _ = self._predictor.predict(point_coords=input_points,
                                                  point_labels=input_labels,
                                                  multimask_output=False)
            
        return masks


    def segment_detection(self, img_p: Union[str,np.ndarray], detections: sv.Detections):
        boxes = detections.xyxy.astype(np.int32)
        return self.segment_boxes(img_p, boxes)


    def segment_boxes(self, img_p: Union[str,np.ndarray], boxes: np.ndarray):
        if isinstance(img_p, str):
            img = cv2.imread(img_p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = img_p
        if len(boxes) == 0:
            return np.array([])
        if not isinstance(boxes, np.ndarray):
            boxes = np.array(boxes)

        # If too much boxes (default 300), break in batches because it can overhead memory easily otherwise
        MASK_BATCH = 200

        MAX_IMG_SIZE = (640,640)
        with torch.no_grad():
            # avoid overheading memory by only getting a slice of the image to segment the mask
            if img.shape[0] > MAX_IMG_SIZE[0] or img.shape[1] > MAX_IMG_SIZE[1]:
                img_cuts = [] # list of [(initrow, finalrow), (inticol, finalcol)]
                box_cuts = [] # list of (initidx, finalidx)
                for row in range(0, img.shape[0], MAX_IMG_SIZE[0]):
                    for col in range(0, img.shape[1], MAX_IMG_SIZE[1]):
                        initrow = row
                        finalrow = min(row+MAX_IMG_SIZE[0], img.shape[0])
                        initcol = col
                        finalcol = min(col+MAX_IMG_SIZE[1], img.shape[1])
                        img_cuts.append([(initrow, finalrow), (initcol, finalcol)])

                        BOX_X_TOLERANCE = 20
                        BOX_Y_TOLERANCE = 20
                        box_cut = np.where(
                            (boxes[:,0] >= initrow) & 
                            (boxes[:,1] >= initcol) & 
                            (boxes[:,2] < (finalrow+BOX_X_TOLERANCE)) & 
                            (boxes[:,3] < (finalcol+BOX_Y_TOLERANCE))
                        )
                        box_cuts.append(box_cut)
                masks = []
                for img_cut, box_cut in zip(img_cuts, box_cuts):
                    (initrow,finalrow),(initcol,finalcol) = img_cut
                    cut = img[initrow:finalrow,initcol:finalcol]
                    cut_boxes = np.array([boxes[i] for i in box_cut])
                    self._predictor.set_image(cut)
                    if len(cut_boxes) < MASK_BATCH:
                        cut_masks, _, _ = self._predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=cut_boxes,
                            multimask_output=False,
                        )
                    else:
                        cut_masks = []
                        for i in range(0, len(cut_boxes), MASK_BATCH):
                            batchmasks, _, _ = self._predictor.predict(
                                point_coords=None,
                                point_labels=None,
                                box=cut_boxes[i:(i+MASK_BATCH)],
                                multimask_output=False,
                            )
                            cut_masks.extend(batchmasks)
                            torch.cuda.empty_cache()
                        cut_masks = np.array(cut_masks)
                    # fill_left = initcol
                    # fill_top = initrow
                    # fill_bot = img.shape[0] - finalrow
                    # fill_right = img.shape[1] - finalcol
                    # print(f"FILL_LEFT={fill_left}, FILL_RIGHT={fill_right}, FILL_TOP={fill_top}, FILL_BOT={fill_bot}")
                    # masks.extend([
                    #     cv2.copyMakeBorder(mask.astype(np.uint8).reshape[], fill_top, fill_bot, fill_left, fill_right, cv2.BORDER_CONSTANT, None) for mask in cut_masks
                    # ])
                    for mask in cut_masks:
                        maskh, maskw = mask.shape[-2:]
                        masku8 = mask.reshape(maskh,maskw,1).astype(np.uint8)
                        full = np.zeros(shape=(img.shape[0],img.shape[1],1), dtype=np.uint8)
                        full[initrow:finalrow,initcol:finalcol] = masku8
                        fullh, fullw = full.shape[:2]
                        masks.append(full.reshape(1, fullh, fullw).astype(mask.dtype))
                    torch.cuda.empty_cache()
                masks = np.array(masks)

            else:
                self._predictor.set_image(img)
                if len(boxes) < MASK_BATCH:
                    masks, _, _ = self._predictor.predict(point_coords=None,
                                                        point_labels=None,
                                                        box=boxes,
                                                        multimask_output=False)
                else:
                    masks = []
                    for i in range(0, len(boxes), MASK_BATCH):
                        seg_masks, _, _ = self._predictor.predict(point_coords=None,
                                                                point_labels=None,
                                                                box=boxes[i:(i+MASK_BATCH)],
                                                                multimask_output=False)
                        masks.extend(seg_masks)
                        torch.cuda.empty_cache()
                    masks = np.array(masks)

        torch.cuda.empty_cache()
        return masks
    

    def _load_sam2(self, ckpt_path: str, cfg: str, custom_state_dict: str = None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        #torch.autocast(device, dtype=torch.bfloat16).__enter__()
        self._device = device

        sam2_model = build_sam2(cfg, ckpt_path, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        if custom_state_dict:
            predictor.model.load_state_dict(torch.load(custom_state_dict))

        return predictor

    def _sam2masks_to_contours(self, masks: np.ndarray):
        mask_contours = []
        for mask in masks:
            h, w = mask.shape[-2:]
            mask = mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask.reshape(h,w,1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_contours.append(contours)

        # mask_contours comes out as an array of contours
        # each contours comes as an array containing an array
        # which contains every point, but the points are in another array as well
        # so this is some magic to have mask_contours[i] = ithcontour
        fmt_contours = []
        for contours in mask_contours:
            for contour in contours:
                fmt_contours.append([])
                for points in contour:
                    fmt_contours[-1].append(points[0])
        mask_contours = fmt_contours

        for i in range(len(mask_contours)):
            # since we're changing the mask_contours list on the fly
            # it's important to check if we're not indexing out of range
            if i >= len(mask_contours):
                break

            # Discard any contours with 4 or less points
            # since it's very possible that it's just a mistake (creating boxes as seen in segmentations annotations)
            if len(mask_contours[i]) <= 4:
                mask_contours.pop(i)

        return mask_contours


    # empty cache on destruction
    def __del__(self):
        torch.cuda.empty_cache()
