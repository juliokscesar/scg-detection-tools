from abc import ABC
from typing import Callable

import numpy as np
import cv2
import torch
from ultralytics import YOLO
from super_gradients.training import models
from roboflow import Roboflow
import supervision as sv

SUPPORTED_MODEL_TYPES = ["yolov8", "yolonas", "roboflow"]

class BaseDetectionModel(ABC):
    def __init__(self, model_type: str, underlying_model):
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise Exception(f"model_type {model_type} is not supported. Possible options are: {SUPPORTED_MODEL_TYPES}")

        self._model_type = model_type
        self._underlying_model = underlying_model


    @abstractmethod
    def predict(self, img_path: str, confidence: float, overlap: float) -> sv.Detections:
        pass


    @abstractmethod
    def slice_predict(self, 
                      img_path: str,
                      confidence: float,
                      overlap: float,
                      slice_wh: tuple,
                      slice_overlap_ratio: tuple,
                      embed_slice_callback: Callable[[str, np.ndarray, str, np.ndarray], None] = None) -> sv.Detections:
        def sv_slice_callback(image: np.ndarray) -> sv.Detections:
            # TODO: temp path in utils
            tmpfile = "TODO"
            with open(tmpfile, "wb") as f:
                sliceimg = image.copy()
                cv2.imwrite(f.name, cv2.cvtColor(sliceimg, cv2.COLOR_RGB2BGR))

                det = self.predict(img_path=f.name,
                                   confidence=confidence,
                                   overlap=overlap)

                if embed_slice_callback is not None:
                    embed_slice_callback(img_path, sliceimg, tmpfile, det.xyxy.astype(np.int32))

                return det

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        slicer = sv.InferenceSlicer(callback=sv_slice_callback,
                                    slice_wh=slice_wh,
                                    overlap_ratio_wh=slice_overlap_ratio)
        sliced_detections = slicer(image=img)

        #TODO clear temp folder
        return sliced_detections


class YOLOv8(BaseDetectionModel):
    def __init__(self, yolov8_ckpt_path: str):
        yolo_model = YOLO(yolov8_ckpt_path)
        super().__init__(model_type="yolov8", underlying_model=yolo_model)


    def predict(self, img_path: str, confidence: float, overlap: float) -> sv.Detections:
        results = self._underlying_model.predict(img_path,
                                                 imgsz=640,
                                                 conf=confidence / 100.0,
                                                 iou=overlap / 100.0,
                                                 max_det=1500)
        detections = sv.Detections.from_ultralytics(results[0])
        return detections


    def slice_predict(self, 
                      img_path: str,
                      confidence: float,
                      overlap: float,
                      slice_wh: tuple,
                      slice_overlap_ratio: tuple,
                      embed_slice_callback: Callable[[str, np.ndarray, str, np.ndarray], None] = None) -> sv.Detections:
        return super().slice_predict(img_path, confidence, overlap, slice_wh, slice_overlap_ratio, embed_slice_callback)


class YOLO_NAS(BaseDetectionModel):
    def __init__(self, model_arch: str, checkpoint_path: str, classes: list):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model_arch = model_arch
        self._device = device
        self._classes = classes.copy()

        model = models.get(model_arch, num_classes=len(classes), checkpoint_path=checkpoint_path)
        model.to(device)

        super().__init__(model_type="yolonas", underlying_model=model)


    def predict(self, img_path: str, confidence: float, overlap: float) -> sv.Detections:
        results = self._underlying_model.predict(img_path,
                                                 conf=confidence / 100.0,
                                                 iou=overlap / 100.0)
        # DEBUG
        print(f"DEBUG: YOLO-NAS results: {results}")

        detections = sv.Detections.from_yolo_nas(results)
        return detections


    def slice_predict(self, 
                      img_path: str,
                      confidence: float,
                      overlap: float,
                      slice_wh: tuple,
                      slice_overlap_ratio: tuple,
                      embed_slice_callback: Callable[[str, np.ndarray, str, np.ndarray], None] = None) -> sv.Detections:
        return super().slice_predict(img_path, confidence, overlap, slice_wh, slice_overlap_ratio, embed_slice_callback)


class RoboflowModel(BaseDetectionModel):
    def __init__(self, api_key: str, project: str, version: int):
        rf = Roboflow(api_key=api_key)
        proj = rf.workspace().project(project)
        model = proj.version(version).model

        super().__init__(model_type="roboflow", underlying_model=model)


    def predict(self, img_path: str, confidence: float, overlap: float) -> sv.Detections:
        results = self._underlying_model.predict(img_path,
                                                confidence=confidence,
                                                overlap=overlap).json()
        detections = sv.Detections.from_inference(results)
        return detections


    def slice_predict(self, 
                      img_path: str,
                      confidence: float,
                      overlap: float,
                      slice_wh: tuple,
                      slice_overlap_ratio: tuple,
                      embed_slice_callback: Callable[[str, np.ndarray, str, np.ndarray], None] = None) -> sv.Detections:
        return super().slice_predict(img_path, confidence, overlap, slice_wh, slice_overlap_ratio, embed_slice_callback)

