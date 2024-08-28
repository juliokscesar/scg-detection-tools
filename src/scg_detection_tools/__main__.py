import argparse
import os
from pathlib import Path

from scg_detection_tools.models import SUPPORTED_MODEL_TYPES
import scg_detection_tools.models as md
from scg_detection_tools.utils.file_handling import get_all_files_from_paths, detections_to_file
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

    det_parser.add_argument("--save", action="store_true", help="Save image with detections and all detections boxes.")
    det_parser.add_argument("--no-show", action="store_true", dest="no_show", help="Don't plot image with detections")


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
        YN_CLASSES = "leaf"
        model = md.YOLO_NAS(YN_ARCH, model_path, YN_CLASSES)
    elif model_type == "roboflow":
        project = input("Roboflow project: ")
        version = int(input("Version: "))
        api_key = os.getenv("ROBOFLOW_API_KEY")
        model = md.RoboflowModel(api_key, project, version)

    img_files = get_all_files_from_paths(*img_source)
    
    det_params = {
        "confidence": args.confidence,
        "overlap": args.overlap,
        "use_slice": args.slice,
        "slice_wh": (args.slice_w, args.slice_h),
        "slice_overlap_ratio": (args.slice_overlap/100.0, args.slice_overlap/100.0),
        "embed_slice_callback": None,
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
            detections_to_file(out_file=f"det{Path(img).stem}.detections", detections=detection)


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


def main():
    args = parse_args()
    command = args.command
    
    FUNC_COMMAND = {
            "detect": detect,
            "segment": segment,
    }

    if command is None:
        raise RuntimeError("command is required")
    else:
        func = FUNC_COMMAND[command]
        func(args)


if __name__ == "__main__":
    main()

