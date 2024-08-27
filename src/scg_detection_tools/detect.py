import supervision as sv
from typing import Union

from scg_detection_tools.models import BaseDetectionModel, FnEmbedSliceCallback

DEFAULT_DETECTION_PARAMS = {
        "confidence": 50.0,
        "overlap": 50.0,

        "use_slice": False,
        "slice_wh": (640, 640),
        "slice_overlap_ratio": (0.1, 0.1),
        "embed_slice_callback": None
}

class Detector:
    def __init__(self, detection_model: BaseDetectionModel, detection_params = DEFAULT_DETECTION_PARAMS):
        self._det_model = detection_model
        self._det_params = detection_params

    def __call__(self, img):
        return self.detect_objects(img)

    # Returns a list of detections for every image (even for a single image)
    def detect_objects(self, img: Union[list, str], **diff_det_params) -> list:
        for param in diff_det_params:
            if param not in self._det_params:
                raise KeyError(f"Unknown parameter '{param}'")
            self._det_params[param] = diff_det_params[param]

        if isinstance(img, str):
            return self._detect_single_image(img)
        elif isinstance(img, list):
            return self._detect_multiple_images(img)
        else:
            raise ValueError("'img' argument must be either a string or a list of strings")

    def _detect_single_image(self, image_path: str) -> sv.Detections:
        use_slice = self._det_params["use_slice"]
        if use_slice:
            detections = self._det_model.slice_predict(img_path=image_path,
                                                       confidence=self._det_params["confidence"],
                                                       overlap=self._det_params["overlap"],
                                                       slice_wh=self._det_params["slice_wh"],
                                                       slice_overlap_ratio=self._det_params["slice_overlap_ratio"],
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

