from enum import Enum
import os
from pathlib import Path
import logging
import shutil
from typing import Tuple, List, Union
import numpy as np
import yaml
import cv2
import warnings

from scg_detection_tools.utils.file_handling import(
        get_all_files_from_paths, file_exists, read_yaml
)
from scg_detection_tools.utils.image_tools import save_image

DATASET_MODES = ["train", "val", "test"]
# Manage datasets in YOLOv8 format
class Dataset:
    def __init__(self, name: str, dataset_dir: str, classes: List[str] = ["leaf"]):
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
                if mode == "val":
                    mode = "valid"
                os.makedirs(os.path.join(dataset_dir, mode+"/images"))
                os.makedirs(os.path.join(dataset_dir, mode+"/labels"))

    @property
    def directory(self):
        return self._dataset_dir

    def add(self, img_path: str, annotations: list, mode: str = "train"):
        if not os.path.isdir(".temp"):
            os.mkdir(".temp")
        
        # Scale if image is not 640x640
        img = cv2.imread(img_path)
        if img is None:
            warnings.warn(f"Coudn't load image {img_path} to check for it's size. Skipping it")
            return
        imgh, imgw = img.shape[:2]
        if imgh != 640 or imgw != 640:
            img = cv2.resize(img, (640,640))
            scaled_path = f"scaled_{os.path.basename(img_path)}"
            save_image(img=img, name=scaled_path, dir=".temp/", cvt_to_bgr=False, notify_save=False)
            img_path = os.path.join(".temp", scaled_path)
        
        ann_file = ".temp/" + Path(img_path).stem + ".txt"
        with open(ann_file, "w") as f:
            for ann in annotations:
                f.write(f"{int(ann[0])} {' '.join([str(float(x)) for x in ann[1:]])}\n")


        data_img = {"image": img_path, "annotations": ann_file}
        self._data[mode].append(data_img)

    def remove(self, index: Union[str, int], mode: str = "train"):
        if mode not in self._data:
            raise ValueError("Argument 'mode' must 'train', 'val' or 'test'")
        # Get image data index if img is string
        if isinstance(index, str):
            for i in range(len(self._data[mode])):
                if self._data[mode][i]["image"] == index:
                    index = i
                    break
        # check again if index is still a string
        if index >= len(self._data[mode]) or isinstance(index, str):
            print(f"len(self._data[mode])={len(self._data[mode])}, index={index}, type(index)={type(index)}, isint={isinstance(index,int)}")
            raise ValueError("'index' must be either the image full path or the index to that image in the mode data list")
        index_data = self._data[mode][index]

        # Delete image and annotation files if in dataset folder
        for keyfile in index_data:
            if file_exists(os.path.join(self._dataset_dir, os.path.basename(index_data[keyfile]))):
                os.remove(os.path.join(self._dataset_dir, os.path.basename(index_data[keyfile])))
        
        # Finally delete item from self._data and return removed
        return self._data[mode].pop(index)

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
        """ Save stored data to the output directory """
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

                data["image"] = img_path
                data["annotations"] = ann_path

        # Save data.yaml
        with open(os.path.join(self._dataset_dir, "data.yaml"), "w") as f:
            yaml.dump(self._dataset, f)

    def len_data(self, mode: Union[str,None] = None):
        if mode is None:
            return sum([len(self._data[m]) for m in self._data])
        return len(self._data[mode])
    
    def get_data(self, mode="train"):
        """ Return stored data as a list of dictionaries being: {"image": img_path, "annotation": ann_path} """
        return self._data[mode]


    def annotations_of(img_path: str, mode="train"):
        """ Return annotation classes and contours from annotation file """
        ann_file = None
        for data in self._data[mode]:
            if data["image"] == img_path:
                ann_file = data["annotations"]
                break

        if ann_file is None:
            raise ValueError(f"{img_path} not in dataset's mode '{mode}'")
        
        return read_dataset_annotation(ann_file)

    def split_modes(self, src="train", dest="val", ratio=0.1):
        """ Split images from one mode to another. The amount to be transfered is ratio*src_amount. The images are randomly selected """
        n_transfer = int(len(self._data[src]) * ratio) + 1
        
        rng = np.random.default_rng()
        for _ in range(n_transfer):
            trans_idx = rng.integers(len(self._data[src]))

            data = self._data[src].pop(trans_idx)
        
            # delete image from mode 'src' folder if it exists
            img_dir = os.path.join(self._dataset_dir, self._dataset[src])
            img_src = os.path.join(img_dir, os.path.basename(data["image"]))
            if file_exists(img_src):
                tmp_path = f".temp/{os.path.basename(img_src)}"
                shutil.copy(img_src, dst=tmp_path)
                os.remove(img_src)
                data["image"] = tmp_path
            
            # same for annotation
            ann_dir = img_dir[:-6] + "labels"
            ann_src = os.path.join(ann_dir, os.path.basename(data["annotations"]))
            if file_exists(ann_src):
                tmp_path = f".temp/{os.path.basename(ann_src)}"
                shutil.copy(ann_src, tmp_path)
                os.remove(ann_src)
                data["annotations"] = tmp_path

            self._data[dest].append(data)

    


def read_dataset_annotation(ann_file: str, separate_class=True):
    if not file_exists(ann_file):
        raise FileExistsError(f"File {ann_file} doesn't exist")
    
    annotations = []
    with open(ann_file, "r", encoding="utf-8") as f:
        for line in f:
            data = line.split()
            annotations.append([])
            annotations[-1].append(int(data[0]))
            annotations[-1].extend([float(x) for x in data[1:]])

    if separate_class:
        nclasses = []
        contours = []
        for ann in annotations:
            nclasses.append(ann[0])
            contours.append(ann[1:])
        return (nclasses, contours)

    else:
        return annotations
