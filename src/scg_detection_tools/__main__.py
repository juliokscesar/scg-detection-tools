import argparse

from scg_detection_tools.models import SUPPORTED_MODEL_TYPES

def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    det_parser = subparsers.add_parser("detect", help="Run object detection on an image")
    det_parser.add_argument("img_source", 
                            nargs="*", 
                            default=None,
                            dest="img_source", 
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

    det_parser.add_argument("--save", action="store_true", help="Save image with detections")
    det_parser.add_argument("--no-show", action="store_true", dest="no_show", help="Don't plot image with detections")


    seg_parser = subparsers.add_parser("segment", help="Run instance segmentation on images using SAM2")
    seg_parser.add_argument("img_source",
                            dest="img_source",
                            nargs="*",
                            defaut=None,
                            help="Source of image(s). Can be a single file, a list, or a directory")
    seg_parser.add_argument("--no-yolo-assist",
                            dest="disable_yolo",
                            action="store_true",
                            help="Don't use bounding boxes from YOLO detection")
    seg_parser.add_argument("--no-slice",
                            dest="no_slice",
                            action="store_true",
                            help="Don't use slice detections with YOLO")



def main():
    pass

if __name__ == "__main__":
    main()

