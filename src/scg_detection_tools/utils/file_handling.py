import os
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

def get_all_files_from_paths(*args, skip_ext: List[str] = None):
    files = []
    for path in args:
        if os.path.isfile(path):
            if file_ext(path) in skip_ext:
                continue
            files.append(path)

        elif os.path.isdir(path):
            for (root, _, filenames) in os.walk(path):
                files.extend([os.path.join(root, file) for file in filenames if file_ext(file) not in skip_ext])
        
        else:
            raise RuntimeError(f"{path} is an invalid file source")

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

def read_detection_boxes_file(file: str) -> np.ndarray:
    boxes = []
    with open(file, "r") as f:
        for line in f:
            box = [int(x) for x in line.strip().split()]
            boxes.append(box)
    return boxes

