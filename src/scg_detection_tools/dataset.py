from enum import Enum
import os
import logging

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
        pass

    def load(self):
        dataset_yaml = os.path.join(self._dataset_dir, "data.yaml")
        if not file_exists(dataset_yaml):
            raise RuntimeError(f"Attempting to load dataset {self._name}, but no data.yaml found in {dataset_yaml}")
        
        self._dataset = read_yaml(dataset_yaml)
        for mode in DATASET_MODES:
            img_dir = os.path.join(self._dataset_dir, self._dataset[mode])
            img_paths = get_all_files_from_paths(img_dir)
            if len(img_paths) == 0:
                logging.warning(f"Attempting to load images from {img_dir}, but no images were found")
                continue

            for img in img_paths:
                pass
        

    def save(self):
        pass


