import logging
import supervision as sv
from typing import Union
import cv2
import time
import os

from scg_detection_tools.models import BaseDetectionModel, FnEmbedSliceCallback
from scg_detection_tools.filters import DetectionFilterDuplicates, DetectionFilterSize
from scg_detection_tools.preprocess import ImagePreprocess, ImagePreprocessPipeline
import scg_detection_tools.utils.image_tools as imtools
from scg_detection_tools.utils.file_handling import generete_temp_path

DEFAULT_DETECTION_PARAMS = {
        "confidence": 50.0,
        "overlap": 50.0,

        "slice_detect": False,
        "slice_wh": (640, 640),
        "slice_overlap_ratio": (0.2, 0.2),
        "slice_iou_threshold": 0.3,
        "slice_fill": True,
        "embed_slice_callback": None,

        "detection_filters": {
            "duplicate_filter": False,
            "duplicate_filter_thresh": 0.95,

            "object_size_filter": False,
            "object_size_max_wh": (80,80),
        },

        "enable_image_preprocess": False,
        "image_preprocess": { 
            "apply_to_ratio": 1.0,
            "parameters": {"contrast_ratio": 1.0,"brightness_delta": 0} ,
        },
}

class Detector:
    def __init__(self, detection_model: BaseDetectionModel, detection_params = DEFAULT_DETECTION_PARAMS, specific_det_params=None):
        self._det_model = detection_model
        if detection_params is None:
            detection_params = DEFAULT_DETECTION_PARAMS
        else:
            for key in DEFAULT_DETECTION_PARAMS:
                if key not in detection_params:
                    detection_params[key] = DEFAULT_DETECTION_PARAMS[key]
        self._det_params = detection_params
        
        if self._det_params["enable_image_preprocess"]:
            self._det_preprocess = ImagePreprocessPipeline(preprocess_steps=[
                ImagePreprocess(
                    preproc_func=imtools.apply_contrast_brightness, 
                    **self._det_params["image_preprocess"]["parameters"],
                )
            ])

        if specific_det_params is not None:
            for spec_det_param in specific_det_params:
                self._det_params[spec_det_param] = specific_det_params[spec_det_param]

    def __call__(self, img):
        return self.detect_objects(img)

    def update_parameters(self, **params):
        for param in params:
            assert(param in DEFAULT_DETECTION_PARAMS)
            if isinstance(params[param], dict):
                for key in params[param]:
                    assert(key in self._det_params[param])
                    self._det_params[param][key] = params[param][key]
            else:
                self._det_params[param] = params[param]

    # Returns a list of detections for every image (even for a single image)
    def detect_objects(self, img: Union[list, str], **diff_det_params) -> list:
        for param in diff_det_params:
            if param not in self._det_params:
                raise KeyError(f"Unknown parameter '{param}'")
            self._det_params[param] = diff_det_params[param]

        if isinstance(img, str):
            return [self._detect_single_image(img)]
        elif isinstance(img, list):
            return self._detect_multiple_images(img)
        else:
            raise ValueError("'img' argument must be either a string or a list of strings")

    def _detect_single_image(self, image_path: str) -> sv.Detections:
        if self._det_params["enable_image_preprocess"]:
            proc = self._det_preprocess(image_path)
            temp = generete_temp_path(suffix=os.path.splitext(image_path)[1])
            imtools.save_image(proc, temp, cvt_to_bgr=True)
            image_path = proc

        slice_detect = self._det_params["slice_detect"]
        if slice_detect:
            detections = self._det_model.slice_predict(
                img_path=image_path,
                confidence=self._det_params["confidence"],
                overlap=self._det_params["overlap"],
                slice_wh=self._det_params["slice_wh"],
                slice_overlap_ratio=self._det_params["slice_overlap_ratio"],
                slice_iou_threshold=self._det_params["slice_iou_threshold"],
                slice_fill=self._det_params["slice_fill"],
                embed_slice_callback=self._det_params["embed_slice_callback"],
            )
        else:
            detections = self._det_model.predict(
                img_path=image_path,
                confidence=self._det_params["confidence"],
                overlap=self._det_params["overlap"],
            )

        # Apply detection filters
        if self._det_params["detection_filters"]["object_size_filter"]:
            logging.info(f"Starting DetectionFilterSize for image {image_path}")
            start = time.time()
            filter = DetectionFilterSize(
                max_obj_wh=self._det_params["detection_filters"]["object_size_max_wh"],
            )
            detections = filter(detections)
            logging.info(f"Ended DetectionFilterSize for image {image_path}. Time of execution: {time.time()-start}")
        if self._det_params["detection_filters"]["duplicate_filter"]:
            logging.info(f"Starting DetectionFilterDuplicates for image {image_path}")
            start = time.time()
            filter = DetectionFilterDuplicates(
                intersection_tresh=self._det_params["detection_filters"]["duplicate_filter_thresh"],
                imghw=cv2.imread(image_path).shape[:2],
            )
            detections = filter(detections)
            logging.info(f"Ended DetectionFilterDuplicates for image {image_path}. Time of execution: {time.time()-start}")
        
        return detections


    def _detect_multiple_images(self, imgs: list):
        all_detections = []
        for img in imgs:
            if not isinstance(img, str):
                raise ValueError("All values passed to list 'imgs' must be a string")
            
            detections = self._detect_single_image(img)
            all_detections.append(detections)

        return all_detections

