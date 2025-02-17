import argparse
from pathlib import Path
import os

from scg_detection_tools.utils.file_handling import (
        read_yaml, get_all_files_from_paths, read_detection_boxes_file, clear_temp_folder
)
from scg_detection_tools.generate import generate_dataset, AugmentationSteps
from scg_detection_tools.models import YOLOv8, YOLO_NAS

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("img_source", nargs="*", help="Source of image(s)")
    
    return parser.parse_args()

def main():
    cfg = read_yaml("autotrain_cfg.yaml")
    dt_cfg = cfg["generate_dataset"]
    train_cfg = cfg["train"]

    args = parse_args()
    
    if len(args.img_source) < 1:
        raise ValueError("Must input at least one image source")

    img_files = get_all_files_from_paths(*args.img_source, skip_ext=[".txt", ".json", ".yaml", ".detections"])
    if len(img_files) == 0:
        raise FileNotFoundError(f"Couldn't retrieve any files from {args.img_source}")

    model_t = cfg["model_type"].strip().lower()
    if model_t == "yolov8":
        model = YOLOv8(yolov8_ckpt_path=cfg["model_path"])
    elif model_t == "yolonas":
        model = YOLO_NAS(cfg["yolonas_arch"], cfg["model_path"], dt_cfg["data_classes"])
    else:
        raise ValueError(f"Model type not supported: {model_t}")


    imgboxes = None
    if dt_cfg["cached_detections"]:
        cache_loc = dt_cfg["cached_detections"]
        cache_files = get_all_files_from_paths(cache_loc, skip_ext=[".png", ".jpeg", ".jpg"])
        imgboxes = {}
        for img in img_files:
            for cache_file in cache_files:
                # if image is img.png, cache file (containing detections xyxy) will be img.png.detections
                if Path(cache_file).stem == os.path.basename(img):
                    imgboxes[img] = read_detection_boxes_file(cache_file)
                    break

    aug_steps = None
    for step in AugmentationSteps:
        if str(step).lower() in dt_cfg["augmentation_steps"]:
            if aug_steps is None:
                aug_steps = step
            else:
                aug_steps |= step

    gen_dt = generate_dataset(name=dt_cfg["data_name"],
                              out_dir=dt_cfg["data_dir"],
                              img_files=img_files,
                              classes=dt_cfg["data_classes"],
                              model=model,
                              sam2_ckpt_path=cfg["sam2_ckpt_path"],
                              sam2_cfg=cfg["sam2_cfg"],
                              use_boxes=dt_cfg["use_boxes"],
                              use_segments=dt_cfg["use_segments"],
                              gen_on_slice=dt_cfg["on_slice"],
                              slice_detect=dt_cfg["slice_detect"],
                              imgboxes_for_segments=imgboxes,
                              augmentation_steps=aug_steps)

    print(f"FINISHED GENERATING DATASET {gen_dt._name}")

    gen_dt.split_modes(src="train", dest="val", ratio=0.1)
    gen_dt.split_modes(src="train", dest="test", ratio=0.05)
    gen_dt.save()

    train_result = model.train(dataset_dir=os.path.abspath(gen_dt.directory),
                               epochs=train_cfg["epochs"],
                               batch=train_cfg["batch"],
                               device=train_cfg["device"],
                               workers=train_cfg["workers"])


    clear_temp_folder()

if __name__ == "__main__":
    main()

