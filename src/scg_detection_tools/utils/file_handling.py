import os
from pathlib import Path
import shutil
import yaml
import supervision as sv
import numpy as np
from typing import List

def generete_temp_path(suffix: str) -> str:
    if not os.path.isdir(".temp"):
        os.mkdir(".temp")
    return os.path.join(".temp", os.urandom(24).hex()+suffix)

def clear_temp_folder():
    try:
        shutil.rmtree(".temp")
    except:
        print("DEBUG: couldn't delete .temp folder")

def file_exists(path: str) -> bool:
    return os.path.isfile(path)

def file_ext(file_path: str) -> str:
    _, ext = os.path.splitext(file_path)
    return ext


def sort_stem(item):
    s = Path(item).stem
    return int(s) if s.isnumeric() else s

def get_all_files_from_paths(*args, skip_ext: List[str] = None, stem_sort=False):
    files = []
    for path in args:
        if os.path.isfile(path):
            if skip_ext is not None:
                if file_ext(path) in skip_ext:
                    continue
            files.append(path)

        elif os.path.isdir(path):
            for (root, _, filenames) in os.walk(path):
                if skip_ext is not None:
                    files.extend([os.path.join(root, file) for file in filenames if file_ext(file) not in skip_ext])
                else:
                    files.extend([os.path.join(root, file) for file in filenames])
        
        else:
            raise RuntimeError(f"{path} is an invalid file source")
    if stem_sort:
        files = sorted(files, key=sort_stem)
    return files


def read_yaml(yaml_file: str):
    content = {}
    with open(yaml_file, "r") as f:
        content = yaml.safe_load(f)

    return content


def detections_to_file(out_file: str, detections: sv.Detections = None, boxes = None):
    with open(out_file, "w") as f:
        if detections is not None:
            boxes = detections.xyxy.astype(np.int32)
        for box in boxes:
            f.write(f"{' '.join([str(int(x)) for x in box])}\n")

def read_detection_boxes_file(file: str) -> list:
    boxes = []
    with open(file, "r") as f:
        for line in f:
            box = [int(x) for x in line.strip().split()]
            boxes.append(box)
    return boxes

def read_cached_detections(img_files: str, path: str) -> List[str]:
    cache_loc = path
    cache_files = get_all_files_from_paths(cache_loc, skip_ext=[".png", ".jpeg", ".jpg"])
    imgboxes = {}
    for img in img_files:
        for cache_file in cache_files:
            # if image is img.png, cache file will be img.png.detections
            if Path(cache_file).stem == os.path.basename(img):
                imgboxes[img] = read_detection_boxes_file(cache_file)
                break

    return imgboxes

def get_annotation_files(imgs: List[str], annotations_path: str):
    img_ann = {}
    ann_files = [os.path.join(annotations_path, f) for f in get_all_files_from_paths(annotations_path)]
    for img in imgs:
        ann_match = [file for file in ann_files if Path(file).stem == Path(img).stem]
        if len(ann_match) == 0:
            img_ann[img] = None
        else:
            img_ann[img] = ann_match[0]
    return img_ann
