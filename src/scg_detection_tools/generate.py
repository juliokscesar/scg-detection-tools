from typing import List
import shutil
import cv2

from scg_detection_tools.models import BaseDetectionModel
from scg_detection_tools.detect import Detector
from scg_detection_tools.dataset import Dataset
import scg_detection_tools.utils.cvt as cvt

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
                     imgboxes_for_segments: dict = None):
    if (not use_boxes) and (not use_segments):
        raise ValueError("generate_dataset require either use_boxes or use_segments to be true")

    gen_dataset = Dataset(name=name, dataset_dir=out_dir, classes=classes)
    
    detector = Detector(model, specific_det_params={"use_slice": slice_detect})
    for img in img_files:
        # keep track of current amount of images to correctly name images
        curr_data_len = len(gen_dataset.get_data(mode="train"))

        ######################################################################################

        if use_boxes:
            if gen_on_slice:
                def _save_slice_callback(img_path, slceimg, tmppath, det_boxes):
                    slice_img_path = f".temp/slice_det{curr_data_len}_{os.path.basename(img_path)}"
                    shutil.copyfile(src=tmppath, dst=slice_img_path)

                    slice_ann = annotation_boxes(det_boxes, sliceimg.shape[1::-1])
                    gen_dataset.add(img_path=slice_img_path, annotations=slice_ann)
            else:
                detections = detector.detect_objects(img)[0]
                img_ann = annotation_boxes(detections.xyxy, imgsz=cv2.imread(img).shape[1::-1])
                gen_dataset.add(img_path=img, annotations=img_ann)

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
                gen_dataset.add(img_path=img, annotations=img_ann)
    
        ######################################################################################
        
        # save after processing each image
        gen_dataset.save()
    
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

