import cv2
import numpy as np
import argparse

from scg_detection_tools.dataset import read_dataset_annotation
from scg_detection_tools.utils.file_handling import file_exists
from scg_detection_tools.utils.cvt import contours_to_masks, boxes_to_masks
import scg_detection_tools.utils.image_tools as imtools

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("img", type=str, help="Image to inspect annotations")
    parser.add_argument("annotation", type=str, help="File containing YOLO annotations")
    
    return parser.parse_args()

def display_gui(window_name="Annotations Viewer", img: np.ndarray = None, ann_contours: list = None, ann_nc: list = None):
    if img is None or ann_contours is None or ann_nc is None:
        raise ValueError("Arguments 'img', 'ann_contours' and 'ann_classes' are all required to display GUI")

    POSSIBLE_COLORS = [
        [30, 6, 255],
        [255, 128, 197],
        [234, 86, 69],
        [0,0,0],
    ]

    class_colors = {i: POSSIBLE_COLORS[i] for i in range(len(POSSIBLE_COLORS))}
    ALPHA = 0.9

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    highlight = -1
    while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE):
        ann_img = img.copy()
        ann_masks = contours_to_masks(ann_contours, imgsz=img.shape[:2])
        for i in range(len(ann_masks)):
            mask = ann_masks[i]
            nc = ann_nc[i]
            color = class_colors[nc] if i != highlight else [255,255,255]

            print(f"Mask {i} class={nc} color={color}")
            ann_img = imtools.segment_annotated_image(ann_img, mask=mask, color=color, alpha=ALPHA)

        cv2.imshow(window_name, cv2.cvtColor(ann_img, cv2.COLOR_RGBA2BGRA))
        k = cv2.waitKey(0)
        if k == 27: # esc key
            cv2.destroyAllWindows()
            break

        elif k == ord('h'):
            print("Press:\n\t'c' to change a mask's class\n\t'v' to highlight a mask\n\t'p' to print all masks indices, classes and colors")
        elif k == ord('c'):
            idx = int(input("Mask index: "))
            if idx >= len(ann_masks):
                print("Invalid mask index")
                continue
            new_c = int(input(f"New class (from: {[key for key in class_colors]}): "))
            if new_c >= len(POSSIBLE_COLORS):
                print("Invalid class number")
                continue
            print(f"Changing mask {idx} class from {ann_nc[idx]} to {new_c}")
            ann_nc[idx] = new_c
        elif k == ord('v'):
            idx = int(input("Mask index: "))
            if idx >= len(ann_masks):
                print("Invalid mask index")
                continue
            highlight = idx





def main():
    args = parse_args()
    
    img_path = args.img
    ann_file = args.annotation

    nclasses, ann_contours = read_dataset_annotation(ann_file)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    display_gui(img=img, ann_contours=ann_contours, ann_nc=nclasses)


if __name__ == "__main__":
    main()

