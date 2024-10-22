from typing import List, Union, Tuple
import logging
import shutil
import cv2
import scipy
import numpy as np
import random
from enum import Flag, auto
import os
from copy import deepcopy

from scg_detection_tools.models import BaseDetectionModel
from scg_detection_tools.detect import Detector
from scg_detection_tools.dataset import Dataset, read_dataset_annotation
import scg_detection_tools.utils.cvt as cvt
from scg_detection_tools.utils.image_tools import save_image, create_annotation_batch
from scg_detection_tools.utils.file_handling import get_annotation_files, clear_temp_folder


class AugmentationSteps(Flag):
    NONE            = auto()
    BLUR            = auto()
    GRAY            = auto()
    FLIP            = auto()
    ROTATE          = auto()
    SHARPEN         = auto()
    NOISE           = auto()
    SCALED_BATCH    = auto()

    def __str__(self):
        return str(self.name)

class DatasetGenerator:
    def __init__(
            self, 
            img_files: List[str], 
            class_labels: List[str], 
            model: BaseDetectionModel = None,
            dataset_name: str = None,
            dataset_dir: str = None,
            pre_annotations_path: str = None, 
            annotation_type: str = "box",
            sam2_path: str = None,
            sam2_cfg: str = "sam2_hiera_t.yaml",
            detection_parameters: dict = None,
            save_on_slice = False,
            on_slice_resize: Union[Tuple[int,int], None] = None,
            augmentation_steps: AugmentationSteps = None,
            keep_ratio: float = 1.0,
        ):
        self._imgs = img_files
        self._img_annotations = { img: None for img in self._imgs }
        
        if pre_annotations_path is not None:
            self._load_annotations(pre_annotations_path)
        
        annotation_type = annotation_type.strip().lower()
        if annotation_type not in ["box", "segment"]:
            raise ValueError("Generated dataset annotation type must be either 'box' or 'segment'")
        if (annotation_type == "segment") and (sam2_path is None):
            raise ValueError("Generated dataset of type 'segment' requires SAM2 checkpoint path.")

        self._cls_labels = class_labels
        self._detector = Detector(model, detection_params=detection_parameters)
        self._img_detections_cache = { img: None for img in self._imgs }
        self._img_masks_cache = { img: None for img in self._imgs }

        if not 0.0 <= keep_ratio <= 1.0:
            raise ValueError("Argument keep_ratio must be between 0 and 1")

        self._config = {
            "use_pre_annotated": (pre_annotations_path is not None),
            "pre_annotations_type": None,
            "dest_annotation_type": annotation_type,
            "save_on_slice": save_on_slice,
            "on_slice_resize": on_slice_resize,
            "sam2_path": sam2_path,
            "sam2_cfg": sam2_cfg,
            "augmentation_steps": augmentation_steps,
            "detection_parameters": self._detector._det_params,
            "keep_ratio": keep_ratio,
        }

        if dataset_name is None:
            dataset_name = "gen_dataset"
        if dataset_dir is  None:
            dataset_dir = "./gen_dataset"
        self._gen_dataset = Dataset(name=dataset_name, dataset_dir=dataset_dir, classes=class_labels)

    def generate(self, save_on_finish=True):
        if not self._config["use_pre_annotated"]:
            self._annotate_images()
        
        # Add annotated images to dataset
        for img, img_annotations in self._img_annotations.items():
            if img_annotations is None:
                logging.error(f"Dataset Generator on generate: img_annotations for {img} is None")
                continue
            self._gen_dataset.add(img_path=img, annotations=img_annotations)

        if self._config["keep_ratio"] < 1.0:
            rng = np.random.default_rng()
            remove_size = int(self._gen_dataset.len_data(mode="train") * (1.0 - self._config["keep_ratio"]))
            remove_indices = rng.choice(self._gen_dataset.len_data(mode="train"), size=remove_size, replace=False)
            remove_indices = sorted(remove_indices, reverse=True)

            logging.info(f"Drroping {remove_size} images from generated dataset")
            for i in remove_indices:
                self._gen_dataset.remove(i, mode="train")

        if save_on_finish:
            self.save()

    def save(self):
        self._gen_dataset.save()
        clear_temp_folder()

    def _annotate_images(self):
        if self._config["save_on_slice"]:
            slice_cache = {}
            on_slice_resize = self._config["on_slice_resize"]
            if self._config["dest_annotation_type"] == "box":
                def _save_on_slice(img_path, sliceimg, tmppath, det_boxes):
                    nonlocal slice_cache, self
                    slice_path = os.path.join(".temp", f"onslice_{len(slice_cache)}_{os.path.basename(img_path)}")
                    slice_img = sliceimg.copy()
                    if on_slice_resize is not None:
                        # there is no need to rescale boxes because they are in relative coordinates
                        slice_img = cv2.resize(slice_img, on_slice_resize, interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(slice_path, cv2.cvtColor(slice_img, cv2.COLOR_RGB2BGR))
                    slice_annotations = annotation_boxes(det_boxes, imgsz=sliceimg.shape[1::-1])
                    slice_cache[slice_path] = slice_annotations
                self._detector.detect_objects(self._imgs, slice_detect=True, embed_slice_callback=_save_on_slice)
            else: # segments
                from scg_detection_tools.segment import SAM2Segment
                seg = SAM2Segment(
                    sam2_ckpt_path=self._config["sam2_path"], 
                    sam2_cfg=self._config["sam2_cfg"], 
                    detection_assist_model=self._detector._det_model
                )
                for img in self._imgs:
                    slice_wh = self._config["detection_parameters"]["slice_wh"]
                    seg_results = seg.slice_segment_detect(img_path=img, slice_wh=slice_wh)
                    for slice in seg_results["slices"]:
                        tmp_path = slice["path"]
                        slice_path = f".temp/slice_seg{len(slice_cache)}_{os.path.basename(img)}"
                        slice_img = cv2.imread(tmp_path)
                        slicesz = slice_img.shape[1::-1]
                        if on_slice_resize is not None:
                            slice_img = cv2.resize(slice_img, on_slice_resize, interpolation=cv2.INTER_CUBIC)
                        cv2.imwrite(slice_path, cv2.cvtColor(slice_img, cv2.COLOR_RGB2BGR))
                        slice_ann = annotation_contours(slice["contours"], imgsz=slicesz)
                        slice_cache[slice_path] = slice_ann
            self._img_annotations = slice_cache
        else:
            self._run_detections()
            if (self._config["dest_annotation_type"] == "segment") and (self._config["pre_annotations_type"] != "segment"):
                self._segment_on_annotations()


    def _run_detections(self):
        detections = self._detector(self._imgs)
        for det, img in zip(detections, self._imgs):
            self._img_detections_cache[img] = det
            imgsz = cv2.imread(img).shape[1::-1]
            ann_boxes = annotation_boxes(det.xyxy.astype(np.int32), imgsz, det_class_id=det.class_id)
            self._img_annotations[img] = ann_boxes

    def _segment_on_annotations(self):
        from scg_detection_tools.segment import SAM2Segment
        segmentor = SAM2Segment(sam2_ckpt_path=self._config["sam2_path"], sam2_cfg=self._config["sam2_cfg"])

        contour_annotated = {}
        for img, img_det in self._img_detections_cache.items():
            if img_det is None:
                logging.error(f"Generating Dataset: trying to segment objects in image {img} but its detections is None")
                continue
            imgsz = cv2.imread(img).shape[1::-1]
            if self._config["use_pre_annotations"] and (img in self._img_annotations) and (self._img_annotations[img] is not None):
                img_boxes = []
                class_id = []
                for ann in self._img_annotations[img]:
                    class_id.append(ann[0])
                    # convert segment to box if any segmentation found
                    box = ann[1:]
                    if len(box) != 4:
                        box = cvt.segment_to_box(box, normalized=True, imgsz=imgsz[::-1])
                    img_boxes.append(box)
                img_boxes = np.array(boxes)
            else:
                img_boxes = img_det.xyxy.astype(np.int32)
                class_id = img_det.class_id

            img_masks = segmentor.segment_boxes(img, img_boxes)
            self._img_masks_cache[img] = img_masks
            img_contours = segmentor._sam2masks_to_contours(img_masks)

            ann_contours = annotation_contours(img_contours, imgsz, det_class_id=class_id)
            contour_annotated[img] = ann_contours
        else:
            for img, img_ann in self._img_annotations.items():
                imgsz = cv2.imread(img).shape
                if img_ann is None:
                    logging.error(f"Trying to segment from pre loaded annotations of image {img}, but its annotations are None")
                    continue
                boxes = []
                class_id = []
                for ann in img_ann:
                    class_id.append(ann[0])
                    # convert segment to box if any segmentation found
                    box = ann[1:]
                    if len(box) != 4:
                        box = cvt.segment_to_box(box, normalized=True, imgsz=imgsz)
                    boxes.append(box)
                boxes = np.array(boxes)

                img_masks = segmentor.segment_boxes(img, boxes)
                self._img_masks_cache[img] = img_masks
                img_contours = segmentor._sam2masks_to_contours(img_masks)
                ann_contours = annotation_contours(img_contours, imgsz=imgsz[1::-1], det_class_id=class_id)
                contour_annotated[img] = ann_contours

        self._img_annotations = contour_annotated

    def _load_annotations(self, annotations_path: str):
        self._img_annotations = get_annotation_files(self._imgs, annotations_path)
        for img in self._img_annotations:
            ann = read_dataset_annotation(self._img_annotations[img], separate_class=False)
            self._config["pre_annotations_type"] = "box" if len(ann[1:]) == 4 else "segment"
            self._img_annotations[img] = ann
            
def augment_steps(steps: AugmentationSteps, imgs: List[str], annotations: List[List[np.ndarray]], augment_ratio=0.3, blur_sigma=5, sharpen_sigma=5, num_batches=5):
    if AugmentationSteps.NONE in steps:
        return None
    
    aug_size = int(len(imgs) * augment_ratio) 
    idx_choices = np.random.choice(len(imgs), size=aug_size)
    using_imgs = []
    using_annotations = []
    for idx in idx_choices:
        using_imgs.append(imgs[idx])
        using_annotations.append(annotations[idx])
    
    augmentations = []
    if AugmentationSteps.BLUR in steps:
        for img, ann in zip(using_imgs, using_annotations):
            orig = cv2.imread(img)
            blur = cv2.GaussianBlur(orig, (blur_sigma,blur_sigma), 0)
            path = f"blur_{os.path.basename(img)}"
            save_image(blur, name=path, dir=".temp")
            augmentations.append({"path": os.path.join(".temp", path), "annotations": ann})

    if AugmentationSteps.GRAY in steps:
        for img, ann in zip(using_imgs, using_annotations):
            orig = cv2.imread(img)
            gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
            path = f"gray_{os.path.basename(img)}"
            save_image(gray, name=path, dir=".temp")
            augmentations.append({"path": os.path.join(".temp", path), "annotations": ann})

    if AugmentationSteps.FLIP in steps:
        raise NotImplemented()
    if AugmentationSteps.ROTATE in steps:
        raise NotImplemented()
    
    if AugmentationSteps.SHARPEN in steps:
        for img, ann in zip(using_imgs, using_annotations):
            orig = cv2.imread(img)
            blur = cv2.GaussianBlur(orig, (sharpen_sigma, sharpen_sigma), 0)
            sharpen = cv2.addWeighted(blur, 7.5, orig, -6.3, 0)
            path = f"sharpen_{os.path.basename(img)}"
            save_image(sharpen, name=path, dir=".temp")
            augmentations.append({"path": os.path.join(".temp", path), "annotations": ann})
        
    if AugmentationSteps.NOISE in steps:
        for img, ann in zip(using_imgs, using_annotations):
            orig = cv2.imread(img)
            noise = np.random.normal(0, 0.6, orig.shape).astype(np.uint8)
            noisy = cv2.add(orig, noise)
            path = f"noise_{os.path.basename(img)}"
            save_image(noisy, name=path, dir=".temp")
            augmentations.append({"path": os.path.join(".temp", path), "annotations": ann})

    if AugmentationSteps.SCALED_BATCH in steps:
        img_contours = [ann[1:] for ann in using_annotations]
        batches, batch_annotations = create_annotation_batch(imgs=using_imgs, imgsz=(640,640), contours=img_contours, images_per_batch=((len(using_imgs) // num_batches)+1))
        for i, (img_batch, ann_batch) in enumerate(zip(batches, batch_annotations)):
            # batch annotations come with just the contours, so we need to add the class in too
            ann_with_cls = [0] * len(ann_batch)
            for i,ann in enumerate(ann_batch):
                ann_with_cls[i].extend(ann_batch)
            path = f"batch_{i}.png"
            save_image(img_batch, name=path, dir=".temp")
            augmentations.append({"path": os.path.join(".temp", path), "annotations": ann_with_cls})

    return augmentations


def annotation_boxes(det_boxes, imgsz, det_class_id = None):
    fmt_boxes = cvt.detbox_to_yolo_fmt(det_boxes, imgsz)
    ann = []
    for idx, box in enumerate(fmt_boxes):
        if (det_class_id is not None) and (idx < len(det_class_id)):
            ann.append([det_class_id[idx]])
        else:
            ann.append([0])
        ann[-1].extend(box)
    return ann

def annotation_contours(contours, imgsz, det_class_id = None):
    fmt_cnt = cvt.contour_to_yolo_fmt(contours, imgsz)
    ann = []
    for idx, contour in enumerate(fmt_cnt):
        if (det_class_id is not None) and (idx < len(det_class_id)):
            ann.append([det_class_id[idx]])
        else:
            ann.append([0])
        ann[-1].extend(contour)
    return ann


def seg_to_box_dataset(dataset_dir: str):
    dt = Dataset(name=os.path.basename(dataset_dir), dataset_dir=dataset_dir)
    dt.load()
    
    boxdt = Dataset(name=f"{dt._name}_box", dataset_dir=f"{dt.directory}_box")

    modes = ["train", "val", "test"]
    for mode in modes:
        for data in dt.get_data(mode=mode):
            imgsz = cv2.imread(data["image"]).shape[:2]
            ann = read_dataset_annotation(data["annotations"], separate_class=False)
            for i in range(len(ann)):
                points = np.array(ann[i][1:])
                points = points.reshape(len(points)//2, 2)
                box = cvt.segment_to_box(points, normalized=True, imgsz=imgsz)
                fmt_box = cvt.detbox_to_yolo_fmt([box], imgsz)[0]
                new_ann = [ann[i][0]]
                new_ann.extend(fmt_box)
                ann[i] = new_ann

            boxdt.add(data["image"], ann, mode=mode)

    boxdt.save()

