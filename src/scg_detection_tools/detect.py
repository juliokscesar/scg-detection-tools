import supervision as sv
from typing import Union

from scg_detection_tools.models import BaseDetectionModel, FnEmbedSliceCallback

DEFAULT_DETECTION_PARAMS = {
        "confidence": 50.0,
        "overlap": 50.0,

        "slice_detect": False,
        "slice_wh": (640, 640),
        "slice_overlap_ratio": (0.2, 0.2),
        "slice_iou_threshold": 0.3,
        "slice_fill": True,
        "embed_slice_callback": None
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
        
        if specific_det_params is not None:
            for spec_det_param in specific_det_params:
                self._det_params[spec_det_param] = specific_det_params[spec_det_param]

    def __call__(self, img):
        return self.detect_objects(img)

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
        slice_detect = self._det_params["slice_detect"]
        if slice_detect:
            detections = self._det_model.slice_predict(img_path=image_path,
                                                       confidence=self._det_params["confidence"],
                                                       overlap=self._det_params["overlap"],
                                                       slice_wh=self._det_params["slice_wh"],
                                                       slice_overlap_ratio=self._det_params["slice_overlap_ratio"],
                                                       slice_iou_threshold=self._det_params["slice_iou_threshold"],
                                                       slice_fill=self._det_params["slice_fill"],
                                                       embed_slice_callback=self._det_params["embed_slice_callback"])
        else:
            detections = self._det_model.predict(img_path=image_path,
                                                 confidence=self._det_params["confidence"],
                                                 overlap=self._det_params["overlap"])
        return detections


    def _detect_multiple_images(self, imgs: list):
        all_detections = []
        for img in imgs:
            if not isinstance(img, str):
                raise ValueError("All values passed to list 'imgs' must be a string")
            
            detections = self._detect_single_image(img)
            all_detections.append(detections)

        return all_detections

