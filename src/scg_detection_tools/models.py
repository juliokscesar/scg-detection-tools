from abc import ABC, abstractmethod
from typing import Callable, Union

import numpy as np
import cv2
import torch
from ultralytics import YOLO
from super_gradients.training import models
from roboflow import Roboflow
import supervision as sv
from pathlib import Path
import os

from scg_detection_tools.utils.file_handling import generete_temp_path, clear_temp_folder, file_exists

SUPPORTED_MODEL_TYPES = ["yolov8", "yolonas", "roboflow"]

FnEmbedSliceCallback: type = Callable[[str, np.ndarray, str, np.ndarray], None]

class BaseDetectionModel(ABC):
    def __init__(self, model_type: str, model_ckpt_path: str, underlying_model):
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise Exception(f"model_type {model_type} is not supported. Possible options are: {SUPPORTED_MODEL_TYPES}")

        self._model_type = model_type
        self._model_ckpt_path = model_ckpt_path
        self._underlying_model = underlying_model


    @abstractmethod
    def predict(self, img_path: str, confidence: float, overlap: float, box_filter=True, box_filter_iou=0.8) -> sv.Detections:
        pass


    @abstractmethod
    def slice_predict(self, 
                      img_path: str,
                      confidence: float,
                      overlap: float,
                      slice_wh: tuple,
                      slice_overlap_ratio: tuple,
                      slice_iou_threshold: float,
                      slice_fill = True,
                      box_filter=True,
                      box_filter_iou=0.8,
                      embed_slice_callback: Callable[[str, np.ndarray, str, np.ndarray], None] = None) -> sv.Detections:
        def sv_slice_callback(image: np.ndarray) -> sv.Detections:
            # Check if slice is smaller than the desired
            # if so, then fills to the right and to bottom with black pixels
            # to get, but without changing the original pixels coordinates
            sliceimg = image.copy()
            if slice_fill:
                h, w = sliceimg.shape[:2]
                sh, sw = slice_wh
                bot_fill = sh - h
                right_fill = sw - w
                if bot_fill or right_fill:
                    sliceimg = cv2.copyMakeBorder(sliceimg, 0, bot_fill, 0, right_fill, cv2.BORDER_CONSTANT, None, np.zeros(3))

            tmpfile = generete_temp_path(suffix=Path(img_path).suffix)
            with open(tmpfile, "wb") as f:
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
                                    overlap_ratio_wh=slice_overlap_ratio,
                                    iou_threshold=slice_iou_threshold)
        sliced_detections = slicer(image=img)

        return sliced_detections


    @abstractmethod
    def train(self, 
              dataset_dir: str, 
              epochs=30, 
              batch=8, 
              device: Union[list, str, int] = "cpu", 
              workers=6):
        pass


class YOLOv8(BaseDetectionModel):
    def __init__(self, yolov8_ckpt_path: str):
        yolo_model = YOLO(yolov8_ckpt_path)
        self._device = get_opt_device()
        super().__init__(model_type="yolov8", model_ckpt_path=yolov8_ckpt_path, underlying_model=yolo_model)


    def predict(self, img_path: str, confidence: float, overlap: float) -> sv.Detections:
        results = self._underlying_model.predict(img_path,
                                                 imgsz=640,
                                                 conf=confidence / 100.0,
                                                 iou=overlap / 100.0,
                                                 max_det=10000)
        detections = sv.Detections.from_ultralytics(results[0])
        return detections


    def slice_predict(self, 
                      img_path: str,
                      confidence: float,
                      overlap: float,
                      slice_wh: tuple,
                      slice_overlap_ratio: tuple,
                      slice_iou_threshold: float,
                      slice_fill = True,
                      embed_slice_callback: Callable[[str, np.ndarray, str, np.ndarray], None] = None) -> sv.Detections:
        return super().slice_predict(img_path, confidence, overlap, slice_wh, slice_overlap_ratio, slice_iou_threshold, slice_fill, embed_slice_callback)

    def train(self, 
              dataset_dir: str, 
              epochs=30, 
              batch=8, 
              device: Union[list, str, int] = "cpu", 
              workers=6):
        data_yaml = os.path.join(dataset_dir, "data.yaml")
        if not file_exists(data_yaml):
            raise FileExistsError(f"No 'data.yaml' found in dataset directory {dataset_dir}")

        results = self._underlying_model.train(data=data_yaml, imgsz=640, epochs=epochs, batch=batch, device=device, workers=workers, patience=int(0.4*epochs))
        return results


class YOLO_NAS(BaseDetectionModel):
    def __init__(self, model_arch: str, checkpoint_path: str, classes: list):
        self._model_arch = model_arch
        self._classes = classes.copy()

        self._device = get_opt_device()


        model = models.get(model_arch, num_classes=len(classes), checkpoint_path=checkpoint_path)
        model.to(self._device)

        super().__init__(model_type="yolonas", model_ckpt_path=checkpoint_path, underlying_model=model)


    def predict(self, img_path: str, confidence: float, overlap: float) -> sv.Detections:
        results = self._underlying_model.predict(img_path,
                                                 conf=confidence / 100.0,
                                                 iou=overlap / 100.0)
        detections = sv.Detections.from_yolo_nas(results)
        return detections


    def slice_predict(self, 
                      img_path: str,
                      confidence: float,
                      overlap: float,
                      slice_wh: tuple,
                      slice_overlap_ratio: tuple,
                      slice_iou_threshold: float,
                      slice_fill = True,
                      embed_slice_callback: Callable[[str, np.ndarray, str, np.ndarray], None] = None) -> sv.Detections:
        return super().slice_predict(img_path, confidence, overlap, slice_wh, slice_overlap_ratio, slice_iou_threshold, slice_fill, embed_slice_callback)


    def train(self, 
              dataset_dir: str, 
              epochs=30, 
              batch=8, 
              device: Union[list, str, int] = "cpu", 
              workers=6):
        from scg_detection_tools.utils.yolonas_train import train_yolo_nas

        if not file_exists(os.path.join(dataset_dir, "data.yaml")):
            raise FileExistsError(f"No 'data.yaml' found in dataset directory {dataset_dir}")

        num_gpus = 1
        multi_gpu = False
        if isinstance(device, list):
            num_gpus = len(device)
            multi_gpu = True
        elif isinstance(device, str):
            num_gpus = 0

        train_yolo_nas(dataset_dir=dataset_dir,
                       model_arch=self._model_arch,
                       epochs=epochs,
                       batch=batch,
                       workers=workers,
                       multi_gpu=multi_gpu,
                       num_gpus=num_gpus,
                       pretrained_checkpoint_path=self._model_ckpt_path)
        return True



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
                      slice_iou_threshold: float,
                      slice_fill = True,
                      embed_slice_callback: Callable[[str, np.ndarray, str, np.ndarray], None] = None) -> sv.Detections:
        return super().slice_predict(img_path, confidence, overlap, slice_wh, slice_overlap_ratio, slice_iou_threshold, slice_fill, embed_slice_callback)


    def train(self, 
              dataset_dir: str, 
              epochs=30, 
              batch=8, 
              device: Union[list, str, int] = "cpu", 
              workers=6):
        raise Exception("Roboflow model does not support training")


def get_opt_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def from_type(model_type: str, model_path: str, data_classes = ["leaf"], yolonas_arch = "yolo_nas_l"):
    if model_type == "yolov8":
        return YOLOv8(model_path)
    elif model_type == "yolonas":
        return YOLO_NAS(yolonas_arch, model_path, data_classes)
    else:
        raise NotImplemented()
