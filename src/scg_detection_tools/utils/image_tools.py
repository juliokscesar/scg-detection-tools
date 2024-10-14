import supervision as sv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Union, Tuple, List
import os
from copy import deepcopy

from scg_detection_tools.utils.file_handling import get_all_files_from_paths
import scg_detection_tools.utils.cvt as cvt

def apply_image_mask(img: Union[str, np.ndarray], binary_mask: np.ndarray, color: np.ndarray = None, alpha: float = 0.6, bounding_box: np.ndarray = None) -> np.ndarray:
    """ 
    Apply a mask over an image with specified color and alpha value.
    'binary_mask' must have shape (height, width, 1) with 1s where the object is and 0s otherwise.
    If 'color' provided is None, will use a 'deepskyblue' like color.
    If a 'bounding_box' is provided, binary_mask must have same height and width as the bounding box, and the mask will be applied in the location of the bounding box on the image.
    """
    if isinstance(img, str):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    assert(img.shape[2] == 3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    img[:,:,3] = 255

    if color is None:
        color = np.array([255, 0, 255]) # Magenta

    masked_img = deepcopy(img)

    if bounding_box is not None:
        x1, y1, x2, y2 = bounding_box
        boxw = x2 - x1
        boxh = y2 - y1
        maskh = binary_mask.shape[0]
        maskw = binary_mask.shape[1]
        mask = binary_mask.copy()
        # Take only the part surrounded by the box if the mask is bigger than the box.
        if (maskh > boxh) and (maskw > boxw):
            mask = mask[0:boxh,0:boxw]
        elif (boxh > maskh) and (boxw > boxh):
            padh = min(0, boxh - maskh)
            padw = min(0, boxw - maskw)
            print("PADH, PADW:", padh, padw)
            mask = np.pad(mask, ((0,padh), (0,padw)), mode="constant", constant_values=0)
    
        # Apply mask on crop
        img_crop = crop_box_image(masked_img, bounding_box)
        # Make mask RGBA with color and alpha
        mask = mask * 255
        mask = mask * np.concatenate((color, [alpha])).reshape(1, 1, -1)

        alpha_crop = img_crop[:,:,3] / 255.0
        alpha_mk = mask[:,:,3] / 255.0
        
        for c in range(3):
            img_crop[:,:,c] = alpha_mk * mask[:,:,c] + alpha_crop * img_crop[:,:,c] * (1 - alpha_mk)

        masked_img[y1:y2,x1:x2] = img_crop

    else:
        assert(img.shape[:2] == binary_mask.shape[:2])
        mask = binary_mask.copy()
        mask = (mask * 255) * np.concatenate((color, [alpha])).reshape(1,1,-1)
        alpha_img = masked_img[:,:,3] / 255.0
        alpha_mk = mask[:,:,3] / 255.0
        for c in range(3):
            masked_img[:,:,c] = alpha_mk * mask[:,:,c] + alpha_img * masked_img[:,:,c] * (1 - alpha_mk)

    return cv2.cvtColor(masked_img, cv2.COLOR_RGBA2RGB)


def mask_img_alpha(mask: np.ndarray, color: np.ndarray, alpha: float, binary_mask=True) -> np.ndarray:
    if binary_mask or (np.max(mask, axis=0) == 1):
        mask = mask * 255
    mask = mask.astype(np.uint8)
    h, w = mask.shape[:2]
    mask_img = mask.reshape(h,w,1) * np.concatenate((color, [alpha])).reshape(1,1,-1)
    return mask_img

def box_annotated_image(img: Union[str, np.ndarray], boxes: Union[sv.Detections, np.ndarray], box_thickness: int = 1) -> np.ndarray:
    if isinstance(img, str):
        img = cv2.imread(img)
    if isinstance(boxes, sv.Detections):
        boxes = boxes.xyxy.astype(np.int32)
    annotated_img = img.copy()
    BOX_COLOR = [255, 0, 255] # magenta
    for box in boxes:
        x1, y1, x2, y2 = box
        annotated_img = cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color=BOX_COLOR, thickness=box_thickness)
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

def contour_annotated_image(img: Union[str, np.ndarray], contour: np.ndarray, color: np.ndarray, alpha: float, normalized=True) -> np.ndarray:
    if isinstance(img, str):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    mask = cvt.contours_to_masks([contour], imgsz=img.shape[1::-1], normalized=normalized, binary_mask=True)[0]
    return segment_annotated_image(img, mask, color, alpha)


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

def save_image(img: np.ndarray, name: str, dir: Union[str, None] = None, cvt_to_bgr=False, notify_save=False):
    if cvt_to_bgr:
        if img.shape[:-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif img.shape[:-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    if isinstance(dir, str):
        if not os.path.isdir(dir):
            os.makedirs(dir, exist_ok=True)
        out_file = os.path.join(dir, name)
    else:
        out_file = name
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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    row0, col0, row1, col1 = box_xyxy
    return img[col0:col1, row0:row1]

def plot_image_detection(img: Union[str, np.ndarray], boxes: Union[sv.Detections, np.ndarray], box_thickness: int = 1, cvt_to_rgb=True):
    if isinstance(img, str):
        img = cv2.imread(img)
    
    if isinstance(boxes, sv.Detections):
        boxes = boxes.xyxy.astype(np.int32)
    annotated = box_annotated_image(img, boxes, box_thickness=box_thickness)
    plot_image(annotated, cvt_to_rgb=cvt_to_rgb)

def save_image_detection(default_imgpath: str,
                         boxes: Union[sv.Detections, np.ndarray],
                         save_name: str,
                         save_dir: str,
                         box_thicknes: int = 1):
    if isinstance(boxes, sv.Detections):
        boxes = boxes.xyxy.astype(np.int32)
    annotated = box_annotated_image(default_imgpath, boxes, box_thicknes)
    save_image(annotated, save_name, save_dir)


def create_annotation_batch(imgs: Union[List[np.ndarray], List[str]], imgsz: Tuple[int,int], contours: List[np.ndarray],images_per_batch: int):
    assert((len(imgs) > images_per_batch))
    
    ## PREPROCESS IMAGES    
    # STD_IMG_SHAPE = (h, w, c)
    STD_IMG_SHAPE = list(imgsz); STD_IMG_SHAPE.append(3)
    STD_IMG_SHAPE = tuple(STD_IMG_SHAPE)
    for i, img in enumerate(imgs):
        if isinstance(img, str):
            img = cv2.imread(img)
        assert(img is not None)
        # Resize to standard size
        if (img.shape != STD_IMG_SHAPE):
            img = cv2.resize(img, STD_IMG_SHAPE[:2], cv2.INTER_CUBIC)
        imgs[i] = img
    imgs = np.array(imgs)

    # Grid for placing images
    grid_size = int(np.ceil(np.sqrt(images_per_batch)))
    scaled_imgsz = (STD_IMG_SHAPE[0] // grid_size, STD_IMG_SHAPE[1] // grid_size, STD_IMG_SHAPE[2])
    num_batches = int(np.ceil(len(imgs) / images_per_batch))
    batches = np.zeros((num_batches, STD_IMG_SHAPE[0], STD_IMG_SHAPE[1], STD_IMG_SHAPE[2]), dtype=np.uint8)
    batch_contours = []

    for batch_idx in range(0, num_batches):
        batch_imgs = imgs[(batch_idx*images_per_batch):((batch_idx*images_per_batch)+images_per_batch)]
        batch_contours.append([])
        for img_idx, img in enumerate(batch_imgs):
            scaled = cv2.resize(img, scaled_imgsz[:2])
            # Position of the image in the grid
            row = img_idx // grid_size
            col = img_idx % grid_size
            x_offset = col * scaled_imgsz[1]
            y_offset = row * scaled_imgsz[0]
            batches[batch_idx][y_offset:(y_offset+scaled_imgsz[0]), x_offset:(x_offset+scaled_imgsz[1])] = scaled

            # adjust countours
            for contour in contours[(batch_idx*images_per_batch)+img_idx]:
                if len(contour) == 4: # contour is a box
                    x_center, y_center, box_w, box_h = contour
                    # Convert normalized to absolute coordinates
                    x_center *= STD_IMG_SHAPE[1]
                    y_center *= STD_IMG_SHAPE[0]
                    box_w *= STD_IMG_SHAPE[1]
                    box_h *= STD_IMG_SHAPE[0]

                    x_center_scaled = (x_center * scaled_imgsz[1] / STD_IMG_SHAPE[1]) + x_offset
                    y_center_scaled = (y_center * scaled_imgsz[0] / STD_IMG_SHAPE[0]) + y_offset
                    box_w_scaled = box_w * scaled_imgsz[1] / STD_IMG_SHAPE[1]
                    box_h_scaled = box_h * scaled_imgsz[0] / STD_IMG_SHAPE[0]
                    batch_box = np.array(
                        [
                            x_center_scaled / STD_IMG_SHAPE[1], 
                            y_center_scaled / STD_IMG_SHAPE[0], 
                            box_w_scaled / STD_IMG_SHAPE[1], 
                            box_h_scaled / STD_IMG_SHAPE[0],
                        ]
                    )
                    batch_contours[-1].append(batch_box)
                else: # contour is a segmentation contour
                    if not isinstance(contour, np.ndarray):
                        contour = np.array(contour)
                    # contour comes as an array of (Npoints) x0 y0 y1 x1 ... xn yn all normalized
                    contour = contour.reshape(len(contour) // 2, 2) # reshape into (Npoints, 2)
                    for i, (x,y) in enumerate(contour):
                        x_abs = x * STD_IMG_SHAPE[1]
                        y_abs = y * STD_IMG_SHAPE[0]
                        x_s = (x_abs * scaled_imgsz[1] / STD_IMG_SHAPE[1]) + x_offset
                        y_s = (y_abs * scaled_imgsz[0] / STD_IMG_SHAPE[0]) + y_offset
                        x_n = x_s / STD_IMG_SHAPE[1]
                        y_n = y_s / STD_IMG_SHAPE[0]
                        contour[i] = np.array((x_n, y_n))
                    batch_contours[-1].append(contour.flatten())
                    

    return batches, batch_contours


def apply_contrast_brightness(img: Union[str, np.ndarray], contrast_ratio: float = 1.0, brightness_delta: int = 0) -> np.ndarray:
    if isinstance(img, str):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    altered = cv2.convertScaleAbs(img, alpha=contrast_ratio, beta=brightness_delta)
    return altered
