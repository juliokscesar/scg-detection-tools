import supervision as sv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Union

from scg_detection_tools.utils.file_handling import get_all_files_from_paths

def box_annotated_image(default_imgpath: str, detections: sv.Detections, box_thickness: int = 1) -> np.ndarray:
    box_annotator = sv.BoxAnnotator(thickness=box_thickness)
    default_img = cv2.imread(default_imgpath)

    annotated_img = box_annotator.annotate(scene=default_img.copy(),
                                           detections=detections)
    return annotated_img

def segment_annotated_image(default_imgpath: str, masks: np.ndarray) -> np.ndarray:
    img = cv2.imread(default_imgpath)
    masked_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    masked_img[:,:,3] = 1.0

    mask_color = np.array([30/255, 144/255, 255/255, 1.0])
    for mask in masks:
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_img = mask.reshape(h,w,1) * mask_color.reshape(1,1,-1)

        alpha_mask = mask_img[:,:,3]
        alpha_dest = 1.0 - alpha_mask

        for c in range(3):
            masked_img[:,:,c] = (alpha_mask * mask_img[:,:,c] + alpha_dest * masked_img[:,:,c])
    

    return masked_img

def plot_image(img: np.ndarray, cvt_to_rgb=True):
    if img.ndim == 2:
        plt.imshow(img, cmap="gray")
    elif cvt_to_rgb:
        if img.shape[-1] == 4:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGBA2RGB))
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img)

    plt.axis("off")
    plt.show()

def save_image(img: np.ndarray, name: str, dir: str = "exp", cvt_to_bgr=False, notify_save=True):
    if cvt_to_bgr:
        if img.shape[:-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif img.shape[:-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    out_file = f"{dir}/{name}"
    cv2.imwrite(out_file, img)
    
    if notify_save:
        print(f"Saved image {out_file}")

def load_images(*args, color_space="RGB"):
    files = get_all_files_from_paths(*args)
    imgs = []
    for file in files:
        img = cv2.imread(file)
        if color_space == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif color_space == "RGBA":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        else:
            raise RuntimeError(f"Color space {color_space} in load_images not implemented yet.")

        imgs.append(img)
    
    return imgs

def crop_box_image(img: Union[np.ndarray, str],
                   box_xyxy: np.ndarray):
    if isinstance(img, str):
        img = cv2.imread(img)

    row0, col0 = box_xyxy[0]
    row1, col1 = box_xyxy[1]
    return img[col0:(col1+1), row0:(row1+1)]


def save_image_detection(default_imgpath: str,
                         detections: sv.Detections,
                         save_name: str,
                         save_dir: str,
                         box_thicknes: int = 1):
    annotated = box_annotated_image(default_imgpath, detections, box_thicknes)
    save_image(annotated, save_name, save_dir)

