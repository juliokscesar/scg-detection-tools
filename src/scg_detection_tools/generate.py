from typing import List
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
from scg_detection_tools.utils.image_tools import save_image


class AugmentationSteps(Flag):
    BLUR        = auto()
    GRAY        = auto()
    FLIP        = auto()
    ROTATE      = auto()
    SHARPEN     = auto()
    NOISE       = auto()

    def __str__(self):
        return str(self.name)


def generate_dataset(name: str, 
                     out_dir: str,
                     img_files: List[str],
                     classes: List[str],
                     model: BaseDetectionModel,
                     sam2_ckpt_path: str = None,
                     sam2_cfg: str = None,
                     use_boxes=False,
                     use_segments=False,
                     gen_on_slice=False,
                     slice_detect=False,
                     imgboxes_for_segments: dict = None,
                     augmentation_steps: AugmentationSteps = None):
    if (not use_boxes) and (not use_segments):
        raise ValueError("generate_dataset require either use_boxes or use_segments to be true")

    gen_dataset = Dataset(name=name, dataset_dir=out_dir, classes=classes)
    
    detector = Detector(model, specific_det_params={"use_slice": slice_detect})
    for img in img_files:
        # keep track of current amount of images to correctly name images
        curr_data_len = len(gen_dataset.get_data(mode="train"))

        final_img = None
        final_ann = None
        ######################################################################################

        if use_boxes:
            if gen_on_slice:
                def _save_slice_callback(img_path, sliceimg, tmppath, det_boxes):
                    slice_img_path = f".temp/slice_det{curr_data_len}_{os.path.basename(img_path)}"
                    shutil.copyfile(src=tmppath, dst=slice_img_path)

                    slice_ann = annotation_boxes(det_boxes, sliceimg.shape[1::-1])
                    gen_dataset.add(img_path=slice_img_path, annotations=slice_ann)
                
                detector.detect_objects(img, embed_slice_callback=_save_slice_callback)
            else:
                detections = detector.detect_objects(img)[0]
                img_ann = annotation_boxes(detections.xyxy, imgsz=cv2.imread(img).shape[1::-1])
                final_img = img
                final_ann = img_ann


        ######################################################################################

        if use_segments:
            if sam2_ckpt_path is None or sam2_cfg is None:
                raise ValueError("Must provide sam2_ckpt_path and sam2_cfg arguments for generating on segments")

            from scg_detection_tools.segment import SAM2Segment

            seg = SAM2Segment(sam2_ckpt_path=sam2_ckpt_path,
                              sam2_cfg=sam2_cfg,
                              detection_assist_model=model)
            
            if gen_on_slice:
                seg_results = seg.slice_segment_detect(img_path=img, slice_wh=(640,640))
                for slice in seg_results["slices"]:
                    tmp_path = slice["path"]
                    slice_path = f".temp/slice_seg{curr_data_len}_{os.path.basename(img)}"
                    slice_ann = annotation_contours(slice["contours"], imgsz=(640,640))

                    gen_dataset.add(img_path=slice_path, annotations=slice_ann)
            else:
                if imgboxes_for_segments is not None:
                    masks = seg._segment_boxes(img_p=img, boxes=imgboxes_for_segments[img])
                    contours = seg._sam2masks_to_contours(masks)
                else:
                    _, contours = seg.detect_segment(img, use_slice_detection=slice_detect)

                img_ann = annotation_contours(contours, imgsz=cv2.imread(img).shape[1::-1])

                final_img = img
                final_ann = img_ann
    

        ######################################################################################

        gen_dataset.save()
    
        if not gen_on_slice:
            gen_dataset.add(img_path=final_img, annotations=final_ann)

        ## Augmentation steps
        if augmentation_steps is not None:
            curr_data = deepcopy(gen_dataset._data["train"])
            for data in curr_data:
                img_path = data["image"]
                img_ann = data["annotations"]
                
                img_class, img_ann = read_dataset_annotation(img_ann)
                if len(img_class) == 0:
                    img_ann = []
                else:
                    ann_with_class = []
                    for i,nc in enumerate(img_class):
                        ann_with_class.append([])
                        ann_with_class[-1].append(nc)
                        ann_with_class[-1].extend(img_ann[i])
                    img_ann = ann_with_class

                orig_img = cv2.imread(img_path)
                base_name = os.path.basename(img_path)

                augmented = []

                if AugmentationSteps.BLUR in augmentation_steps:
                    sigma = random.randrange(3, 11+1, 2)
                    blurred = cv2.GaussianBlur(orig_img, (sigma,sigma), 0)
                    path = f"blur_{base_name}"
                    augmented.append({ "path": os.path.join(".temp", path), "annotations": img_ann })
                
                if AugmentationSteps.GRAY in augmentation_steps:
                    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
                    path = f"gray_{base_name}"
                    save_image(gray, name=path, dir=".temp")
                    augmented.append({ "path": os.path.join(".temp", path), "annotations": img_ann })

                if AugmentationSteps.FLIP in augmentation_steps:
                    raise NotImplemented()
                    # TODO: make transformations in annotations too
                    flip = np.flipud(orig_img)
                    path = f"flip_{base_name}"
                    save_image(flip, name=path, dir=".temp")
                    augmented.append({ "path": os.path.join(".temp", path), "annotations": img_ann })
                
                if AugmentationSteps.ROTATE in augmentation_steps:
                    raise NotImplemented()
                    # TODO: make transformations in annotations too
                    ang = random.randint(30, 270+1)
                    rot = scipy.ndimage.rotate(orig_img, ang, reshape=False)
                    path = f"rotate_{base_name}"
                    save_image(rot, name=path, dir=".temp")
                    augmented.append({ "path": os.path.join(".temp", path), "annotations": img_ann })


                if AugmentationSteps.SHARPEN in augmentation_steps:
                    sigma = random.randrange(3, 7+1, 2)
                    blurred = cv2.GaussianBlur(orig_img, (sigma,sigma), 0)
                    sharpened = cv2.addWeighted(blurred, 7.5, orig_img, -6.3, 0)
                    path = f"sharpen_{base_name}"
                    save_image(sharpened, name=path, dir=".temp")
                    augmented.append({ "path": os.path.join(".temp", path), "annotations": img_ann })

                
                if AugmentationSteps.NOISE in augmentation_steps:
                    rng = np.random.default_rng()
                    noise = rng.normal(0, 0.6, orig_img.shape).astype(np.uint8)
                    noisy = cv2.add(orig_img, noise)
                    path = f"noise_{base_name}"
                    save_image(noisy, name=path, dir=".temp")
                    augmented.append({ "path": os.path.join(".temp", path), "annotations": img_ann })

                for aug in augmented:
                    gen_dataset.add(img_path=aug["path"], annotations=aug["annotations"])

        gen_dataset.save()

        ################################### END OF IMGS LOOP #################################

    
    return gen_dataset
            

def annotation_boxes(det_boxes, imgsz):
    fmt_boxes = cvt.detbox_to_yolo_fmt(det_boxes, imgsz)
    ann = []
    for box in fmt_boxes:
        ann.append([0])
        ann[-1].extend(box)
    return ann

def annotation_contours(contours, imgsz):
    fmt_cnt = cvt.contour_to_yolo_fmt(contours, imgsz)
    ann = []
    for contour in fmt_cnt:
        ann.append([0])
        ann[-1].extend(contour)
    return ann


