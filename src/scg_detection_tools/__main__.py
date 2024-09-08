import argparse
import os
import shutil
from pathlib import Path
import cv2
import warnings
import torch

from scg_detection_tools.models import SUPPORTED_MODEL_TYPES
import scg_detection_tools.models as md
from scg_detection_tools.utils.file_handling import (
        get_all_files_from_paths, detections_to_file, read_cached_detections
)
from scg_detection_tools.utils.image_tools import (
        box_annotated_image, segment_annotated_image, plot_image, save_image
)

def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    det_parser = subparsers.add_parser("detect", help="Run object detection on an image")
    det_parser.add_argument("img_source", 
                            nargs="*", 
                            help="Source of image file(s). Can be a list of paths, a directory, etc.")
    det_parser.add_argument("--model-type",
                            choices=SUPPORTED_MODEL_TYPES,
                            default="yolonas",
                            dest="model_type",
                            help=f"Detection model type. Default is yolonas")
    det_parser.add_argument("-m",
                            "--model-path",
                            dest="model_path",
                            help="Path to model checkpoint")
    det_parser.add_argument("-c",
                            "--confidence",
type=float,
                            default=50.0,
                            help="Confidence parmeter. Default is 50.0")

    det_parser.add_argument("-o", 
                            "--overlap", 
                            type=float, 
                            default="50.0", 
                            help="Overlap parameter. Default is 50.0")

    det_parser.add_argument("-s", 
                            "--slice-detect", 
                            action="store_true", 
dest="slice",
                            help="Use slice detection")
    det_parser.add_argument("--slice-w", 
                            type=int, 
                            dest="slice_w", 
                            default=640,
                            help="Slice width when using slice detection. Default is 640")
    det_parser.add_argument("--slice-h", 
                            type=int, 
                            dest="slice_h", 
                            default=640,
                            help="Slice height when using slice detection. Default is 640")
    det_parser.add_argument("--slice-overlap", 
type=float, 
                            dest="slice_overlap", 
                            default=10.0,
                            help="Slice overlap ratio when using slice detection. Default is 10.0")
    det_parser.add_argument("--slice-iou",
                            type=float,
                            dest="slice_iou",
                            default=40.0,
                            help="Slice IOU threshold to filter duplicate detections")

    det_parser.add_argument("--save", action="store_true", help="Save image with detections and all detections boxes.")
    det_parser.add_argument("--no-show", action="store_true", dest="no_show", help="Don't plot image with detections")
    det_parser.add_argument("--cache-detections", action="store_true", dest="cache_detections", help="Save each box coordinates to a file")


    seg_parser = subparsers.add_parser("segment", help="Run instance segmentation on images using SAM2")
    seg_parser.add_argument("img_source",
                            nargs="*",
                            help="Source of image(s). Can be a single file, a list, or a directory")
    seg_parser.add_argument("-p",
                            "--ckpt-path",
                            dest="ckpt_path",
                            type=str,
                            default=None,
                            help="Path to SAM2 checkpoint")
    seg_parser.add_argument("-c",
                            "--config",
                            type=str,
                            default=None,
                            help="SAM2 checkpoint config")
                            
    seg_parser.add_argument("-y",
                            "--yolo-path",
                            dest="yolo_path",
                            type=str,
                            default=None,
                            help="Path to YOLO(v8,NAS) checkpoint")
    seg_parser.add_argument("--yolov8",
                            action="store_true",
                            help="Use YOLOv8 model instead of YOLO-NAS")
    seg_parser.add_argument("--no-yolo-assist",
                            dest="disable_yolo",
                            action="store_true",
                            help="Don't use bounding boxes from YOLO detection")
    seg_parser.add_argument("--no-slice",
                            dest="no_slice",
                            action="store_true",
                            help="Don't use slice detections with YOLO")
    seg_parser.add_argument("--on-crops",
                            dest="on_crops",
                            action="store_true",
                            help="First crop using YOLO bounding boxes then run segmentation on crops")
    seg_parser.add_argument("--no-show",
                            dest="no_show",
                            action="store_true",
                            help="Don't plot image with the masks")
    seg_parser.add_argument("--save",
                            action="store_true",
                            help="Save image with masks")


    gen_parser = subparsers.add_parser("generate", help="Generate dataset by saving detections/segmentations to annotation files")
    gen_parser.add_argument("img_source",
                            nargs="*",
                            help="Source of images. Can be a single file, multiple, or directory")
    gen_parser.add_argument("--model-type",
                            type=str,
                            default="yolov8",
                            choices=SUPPORTED_MODEL_TYPES,
                            help="Type of object detection model to use")
    gen_parser.add_argument("--model-path",
                            type=str,
                            default=None,
                            help="Path to model's checkpoint")
    gen_parser.add_argument("--sam2-ckpt",
                            dest="sam2_ckpt",
                            type=str,
                            default=None,
                            help="Path to SAM2 checkpoint")
    gen_parser.add_argument("--sam2-cfg",
                            dest="sam2_cfg",
                            type=str,
                            default="sam2_hiera_t.yaml",
                            help="SAM2 config to use if saving segments annotations")
    gen_parser.add_argument("--out-dir",
                            dest="out_dir",
                            type=str,
                            default="gen_out",
                            help="Output directory of generated dataset")
    gen_annotations = gen_parser.add_mutually_exclusive_group(required=True)
    gen_annotations.add_argument("--boxes", action="store_true", help="Save YOLO detection boxes annotations")
    gen_annotations.add_argument("--segments", action="store_true", help="Save SAM2 segments annotations")
    
    gen_parser.add_argument("--on-slice",
                            dest="on_slice",
                            action="store_true",
                            help="Use sliced detection and save annotations of those slices, also copying the slice image to the dataset save directory")
    gen_parser.add_argument("--slice-detect",
                            dest="slice_detect",
                            action="store_true",
                            help="Detect objects using slice detection")
    gen_parser.add_argument("--data-classes",
                            nargs="*",
                            default=["leaf"],
                            help="Classes to use in annotations")
    gen_parser.add_argument("--cached-detections",
                            dest="cached_detections",
                            type=str,
                            default=None,
                            help="Use cached detections (boxes coordinates written to a file)")
    gen_parser.add_argument("--augmentations",
                            nargs="+",
                            type=str,
                            choices=["blur", "gray", "noise", "sharpen"],
                            help="Augmentation steps after processing image")


    train_parser = subparser.add_parser("train", help="Train YOLOv8 or YOLO-NAS on custom dataset")
    train_parser.add_argument("model_type", choices=["yolov8", "yolonas"], help="Type of model to train")
    train_parser.add_argument("model_path", type=str, help="Path to pre-trained model checkpoint")
    train_parser.add_argument("dataset_dir", type=str, help="Path to dataset directory")

    train_parser.add_argument("--epochs", type=int, default=10, help="Epochs to train. Default is 10")
    train_parser.add_argument("--batch", type=int, default=4, help="Data batch size. Default is 4")
    train_parser.add_argument("--workers", type=int, default=2, help="Workers to do training. Default is 2")
    train_parser.add_argument("--device", nargs="+", help="Device to train on. Can be 'cpu', 0 to use first CUDA device, 1 for second, 0,1 for both, etc")



    return parser.parse_args()

def detect(args):
    import scg_detection_tools.detect as det
    img_source = args.img_source
    if not img_source:
        raise RuntimeError("img_source is required for detection")

    model_path = args.model_path
    model_type = args.model_type
    assert(model_type in SUPPORTED_MODEL_TYPES)

    if model_type == "yolov8":
        model = md.YOLOv8(model_path)
    elif model_type == "yolonas":
        YN_ARCH = "yolo_nas_l"
        YN_CLASSES = ["leaf"]
        model = md.YOLO_NAS(YN_ARCH, model_path, YN_CLASSES)
    elif model_type == "roboflow":
        project = input("Roboflow project: ")
        version = int(input("Version: "))
        api_key = os.getenv("ROBOFLOW_API_KEY")
        model = md.RoboflowModel(api_key, project, version)

    img_files = get_all_files_from_paths(*img_source, skip_ext=[".txt", ".json", ".detections"])
    
    slice_count = 0
    if args.cache_detections and args.slice:
        if not os.path.isdir("out_cache"):
            os.mkdir("out_cache")
        def _on_slice_save_detections(img_path, sliceimg, tmppath, boxes):
            nonlocal slice_count
            shutil.copyfile(tmppath, f"out_cache/slice_{slice_count}_det{os.path.basename(img_path)}")
            detections_to_file(out_file=f"out_cache/slice_{slice_count}_det{os.path.basename(img_path)}.detections", boxes=boxes)
            slice_count += 1
        embed_slice_callback = _on_slice_save_detections
    else:
        embed_slice_callback = None

    det_params = {
        "confidence": args.confidence,
        "overlap": args.overlap,
        "use_slice": args.slice,
        "slice_wh": (args.slice_w, args.slice_h),
        "slice_overlap_ratio": (args.slice_overlap/100.0, args.slice_overlap/100.0),
        "slice_iou_threshold": args.slice_iou/100.0,
        "embed_slice_callback": embed_slice_callback,
    }
    detector = det.Detector(model, det_params)
    detections = detector.detect_objects(img_files)
    print("img_files:", img_files, "len(det)=", len(detections))
    for img,detection in zip(img_files, detections):
        annotated = box_annotated_image(img, detection)

        if not args.no_show:
            plot_image(annotated)
        if args.save:
            save_image(annotated, name=f"det{os.path.basename(img)}", dir="out")
            if args.cache_detections:
                detections_to_file(out_file=f"out_cache/det{os.path.basename(img)}.detections", detections=detection)


def segment(args):
    import scg_detection_tools.segment as seg
    ckpt_path = args.ckpt_path
    cfg = args.config
    if not ckpt_path or not cfg:
        raise RuntimeError("arguments --ckpt-path and --config are required for segment")

    if args.disable_yolo:
        raise Exception("No yolo assist not implemented yet")
    
    yolo_path = args.yolo_path
    if not yolo_path:
        raise RuntimeError("Segmentation with YOLO requires yolo checkpoint path")

    if args.yolov8:
        model = md.YOLOv8(yolov8_ckpt_path=yolo_path)
    else:
        # TODO: some way to support more classes and architectures in this cli tool
        YN_ARCH = "yolo_nas_l"
        YN_CLASSES = ["leaf"]

        model = md.YOLO_NAS(YN_ARCH, yolo_path, YN_CLASSES)

    img_source = args.img_source
    img_files = get_all_files_from_paths(*img_source)
    if len(img_files) == 0:
        raise RuntimeError("At least one image source is required for segmentation")

    sg = seg.SAM2Segment(sam2_ckpt_path=ckpt_path,
                         sam2_cfg=cfg,
                         detection_assist_model=model)

    for img in img_files:
        masks, contours = sg.detect_segment(img, (not args.no_slice))
        print(f"len(masks)={len(masks)}. len(contours):{len(contours)}")
        annotated = segment_annotated_image(img, masks)
        if not args.no_show:
            plot_image(annotated)
        if args.save:
            save_image(annotated, name=f"seg{os.path.basename(img)}", dir="out")

def generate(args):
    from scg_detection_tools.generate import generate_dataset, AugmentationSteps

    img_files = get_all_files_from_paths(*args.img_source, skip_ext=[".txt", ".detections", ".json"])

    model_t = args.model_type
    if model_t == "yolov8":
        model = md.YOLOv8(yolov8_ckpt_path=args.model_path)
    elif model_t == "yolonas":
        model = md.YOLO_NAS(model_arch="yolo_nas_l", checkpoint_path=args.model_path, classes=args.data_classes)
    else:
        raise RuntimeError(f"Support for model {model_t} not implemented for dataset generation")

    imgboxes = None
    if args.cached_detections:
        imgboxes = read_cached_detections(args.cached_detections)

    aug_steps = None
    if args.augmentations is not None:
        aug_steps = AugmentationSteps(0)
        for aug in AugmentationSteps:
            if str(aug).lower() in args.augmentations:
                aug_steps |= aug
    
    gen_dataset = generate_dataset(name="gen_dataset",
                                   out_dir="gen_out",
                                   img_files=img_files,
                                   classes=args.data_classes,
                                   model=model,
                                   sam2_ckpt_path=args.sam2_ckpt,
                                   sam2_cfg=args.sam2_cfg,
                                   use_boxes=args.boxes,
                                   use_segments=args.segments,
                                   gen_on_slice=args.on_slice,
                                   slice_detect=args.slice_detect,
                                   imgboxes_for_segments=imgboxes,
                                   augmentation_steps=aug_steps)

    print("FINISHED GENERATING DATASET", gen_dataset._name)


def train(args):
    model_t = args.model_type.strip().lower()

    if model_t == "yolov8":
        model = md.YOLOv8(yolov8_ckpt_path=args.model_path)
    elif model_t == "yolonas":
        YN_ARCH = "yolo_nas_l"
        YN_CLASSES = ["leaf"]
        model = md.YOLO_NAS(YN_ARCH, args.model_path, YN_CLASSES)
    else:
        raise ValueError(f"Model type {model_t} not supported for training")

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        if "cpu" in device:
            device = "cpu"
        else:
            device = [int(d) for d in device]


    dataset_dir = os.path.abspath(args.dataset_dir)
    model.train(dataset_dir=dataset_dir,
                epochs=args.epochs,
                batch=args.batch,
                device=device,
                workers=args.workers)


def main():
    args = parse_args()
    command = args.command
    
    FUNC_COMMAND = {
            "detect": detect,
            "segment": segment,
            "generate": generate,
            "train": train,
    }

    if command is None:
        raise RuntimeError("command is required")
    else:
        func = FUNC_COMMAND[command]
        func(args)


if __name__ == "__main__":
    main()

