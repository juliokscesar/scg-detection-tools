import supervision as sv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Union
import os

from scg_detection_tools.utils.file_handling import get_all_files_from_paths

def mask_img_alpha(mask: np.ndarray, color: np.ndarray, alpha: float) -> np.ndarray:
    mask_img = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
    color_img = np.full_like(mask_img, color, dtype=np.uint8)
    alpha_channel = (mask * alpha * 255).astype(np.uint8)
    color_with_alpha = cv2.merge([color_img[:,:,0],
                                  color_img[:,:,1],
                                  color_img[:,:,2],
                                  alpha_channel])
    return color_with_alpha

def box_annotated_image(default_imgpath: str, detections: sv.Detections, box_thickness: int = 1) -> np.ndarray:
    box_annotator = sv.BoxAnnotator(thickness=box_thickness)
    default_img = cv2.imread(default_imgpath)

    annotated_img = box_annotator.annotate(scene=default_img.copy(),
                                           detections=detections)
    return annotated_img

def segment_annotated_image(default_img: Union[str,np.ndarray], mask: np.ndarray, color: np.ndarray, alpha: float) -> np.ndarray:
    if isinstance(default_img, str):
        img = cv2.imread(default_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    else:
        img = default_img.copy()
        if img.shape[-1] < 4 or img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
            img[:,:,3] = 255

    maskimg = mask_img_alpha(mask=mask.astype(np.uint8), color=color, alpha=alpha)

    alpha_im = img[:,:,3] / 255.0
    alpha_mk = maskimg[:,:,3] / 255.0
    
    masked_img = img.copy()
    for c in range(0,3):
        masked_img[:,:,c] = alpha_mk * maskimg[:,:,c] + alpha_im * masked_img[:,:,c] * (1 - alpha_mk)
    
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

def save_image(img: np.ndarray, name: str, dir: str = "exp", cvt_to_bgr=False, notify_save=False):
    if cvt_to_bgr:
        if img.shape[:-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif img.shape[:-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    if not os.path.isdir(dir):
        os.makedirs(dir, exist_ok=True)

    out_file = os.path.join(dir, name)
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

    print(f"DEBUG image_tools.py: box={box_xyxy}")
    row0, col0, row1, col1 = box_xyxy
    return img[col0:(col1+1), row0:(row1+1)]


def save_image_detection(default_imgpath: str,
                         detections: sv.Detections,
                         save_name: str,
                         save_dir: str,
                         box_thicknes: int = 1):
    annotated = box_annotated_image(default_imgpath, detections, box_thicknes)
    save_image(annotated, save_name, save_dir)

