from enum import Enum
import os
from pathlib import Path
import logging
import shutil
from typing import Tuple, List
import numpy as np

from scg_detection_tools.utils.file_handling import(
        get_all_files_from_paths, file_exists, read_yaml
)

DATASET_MODES = ["train", "val", "test"]
# Manage datasets in YOLOv8 format
class Dataset:
    def __init__(self, name: str, dataset_dir: str, classes: list = ["leaf"]):
        self._dataset = {
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": len(classes),
            "names": classes,
        }

        self._data = {"train": [], "val": [], "test": []}
        self._name = name

        self._dataset_dir = dataset_dir
        if not os.path.isdir(dataset_dir):
            for mode in DATASET_MODES:
                os.makedirs(os.path.join(dataset_dir, mode+"/images"))
                os.makedirs(os.path.join(dataset_dir, mode+"/labels"))

    @property
    def directory():
        return self._dataset_dir

    def add(self, img_path: str, annotations: list, mode: str = "train"):
        if not os.path.isdir(".temp"):
            os.mkdir(".temp")
        
        ann_file = ".temp/" + Path(img_path).stem + ".txt"
        with open(ann_file, "w") as f:
            for ann in annotations:
                f.write(f"{ann[0]} {' '.join([str(x) for x in ann[1:]])}")
        
        data_img = {"image": img_path, "annotations": ann_file}
        self._data[mode].append(data_img)

    def load(self):
        dataset_yaml = os.path.join(self._dataset_dir, "data.yaml")
        if not file_exists(dataset_yaml):
            raise RuntimeError(f"Attempting to load dataset {self._name}, but no data.yaml found in {dataset_yaml}")
        
        self._dataset = read_yaml(dataset_yaml)
        for mode in DATASET_MODES:
            img_dir = os.path.join(self._dataset_dir, self._dataset[mode])
            ann_dir = img_dir[:-6] + "labels"
            img_paths = get_all_files_from_paths(img_dir)
            if len(img_paths) == 0:
                logging.warning(f"Attempting to load images from {img_dir}, but no images were found")
                continue

            for img in img_paths:
                ann_file = os.path.join(ann_dir, Path(img).stem + ".txt")
                data_img = {"image": img, "annotations": ""}
                if file_exists(ann_file):
                    data_img["annotations"] = ann_file

                self._data[mode].append(data_img)


    def save(self):
        for mode in DATASET_MODES:
            img_dir = os.path.join(self._dataset_dir, self._dataset[mode])
            ann_dir = img_dir[:-6] + "labels"

            for data in self._data[mode]:
                img_path = os.path.join(img_dir, os.path.basename(data["image"]))
                if not file_exists(img_path):
                    shutil.copyfile(data["image"], img_path)

                ann_path = os.path.join(ann_dir, os.path.basename(data["annotations"]))
                if not file_exists(ann_path):
                    shutil.copyfile(data["annotations"], ann_path)



def read_dataset_annotation(ann_file: str) -> Tuple[int, np.ndarray]:
    if not file_exists(ann_file):
        raise FileExistsError(f"File {ann_file} doesn't exist")
    
    nclass = -1
    points = []
    with open(ann_file, "r") as f:
        for line in f:
            data = line.split()
            nclass = int(data[0])
            points = [float(x) for x in data[1:]]

    return (nclass, np.array(points))
