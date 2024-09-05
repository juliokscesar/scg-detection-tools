import torch
from super_gradients.training import models
from super_gradients import Trainer
from super_gradients.training.dataloaders.dataloaders import (
        coco_detection_yolo_format_train, coco_detection_yolo_format_val)
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

import argparse

from scg_detection_tools.utils.file_handling import read_yaml

def train_yolo_nas(dataset_dir: str,
                   model_arch: str = "yolo_nas_l",
                   batch: int = 8,
                   epochs: int = 25,
                   workers: int = 2,
                   multi_gpu = False,
                   num_gpus = 1,
                   pretrained_checkpoint_path: str = None,
                   checkpoint_out_dir: str = f"yolonas_trainings",
                   experiment_name: str = "yolonas_train"):

    trainer = Trainer(experiment_name=experiment_name, ckpt_root_dir=checkpoint_out_dir)

    classes = read_yaml(f"{dataset_dir}/data.yaml")["names"]

    dataset_params = {
        "data_dir": dataset_dir,
        "train_images_dir": "train/images",
        "train_labels_dir":"train/labels",
        "val_images_dir":"valid/images",
        "val_labels_dir":"valid/labels",
        "test_images_dir":"test/images",
        "test_labels_dir":"test/labels",
        "classes": classes,
    }

    train_data = coco_detection_yolo_format_train(
        dataset_params={
            "data_dir": dataset_params["data_dir"],
            "images_dir": dataset_params["train_images_dir"],
            "labels_dir": dataset_params["train_labels_dir"],
            "classes": dataset_params["classes"]
        },
        dataloader_params={
            "batch_size": batch,
            "num_workers": workers
        }
    )
    val_data = coco_detection_yolo_format_val(
        dataset_params={
            "data_dir": dataset_params["data_dir"],
            "images_dir": dataset_params["val_images_dir"],
            "labels_dir": dataset_params["val_labels_dir"],
            "classes": dataset_params["classes"]
        },
        dataloader_params={
            "batch_size": batch,
            "num_workers": workers
        }
    )
    test_data = coco_detection_yolo_format_val(
        dataset_params={
            "data_dir": dataset_params["data_dir"],
            "images_dir": dataset_params["test_images_dir"],
            "labels_dir": dataset_params["test_labels_dir"],
            "classes": dataset_params["classes"]
        },
        dataloader_params={
            "batch_size": batch,
            "num_workers": workers
        }
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not pretrained_checkpoint_path:
        model = models.get(model_arch, num_classes=len(dataset_params["classes"]), pretrained_weights="coco").to(device)
    else:
        model = models.get(model_arch, 
                           num_classes=len(dataset_params["classes"]), 
                           checkpoint_path=pretrained_checkpoint_path).to(device)

    train_params = {
        "average_best_models": False,
        "warmup_mode": "linear_epoch_step",
        "warmup_initial_lr": 1e-6,
        "lr_warmup_epochs": 3,
        "initial_lr": 5e-4,
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": 0.1,
        "optimizer": "Adam",
        "optimizer_params": {"weight_decay": 0.0001},
        "zero_weight_decay_on_bias_and_bn": True,
        "ema": True,
        "ema_params": {"decay": 0.9, "decay_type": "threshold"},
        "max_epochs": epochs,
        "mixed_precision": True,
        "loss": PPYoloELoss(
            use_static_assigner=False,
            num_classes=len(dataset_params['classes']),
            reg_max=16
            ),
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=len(dataset_params['classes']),
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7
                    )
                )
            ],
        "metric_to_watch": 'mAP@0.50'
     }

    trainer.train(model=model,
                  training_params=train_params,
                  train_loader=train_data,
                  valid_loader=val_data)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_dir", help="Directory containing datset")

    parser.add_argument("--batch", type=int, default=8, help="Batch size. Default is 8")
    parser.add_argument("--epochs", type=int, default=25, help="Epochs. Default is 25")
    parser.add_argument("--workers", type=int, default=2, help="Workers. Default is 2")
    parser.add_argument("--multi-gpu", dest="multi_gpu", action="store_true", help="Use multi GPU traning")
    parser.add_argument("--num-gpus", dest="num_gpus", type=int, default=1, help="Number of GPUs to use when training. Default uses 1")

    parser.add_argument("--yolonas-arch", dest="model_arch", type=str, default="yolo_nas_l", help="YOLO-NAS architecture (yolo_nas_{s,m,l}). Default is yolo_nas_l")

    parser.add_argument("--model-path", dest="model_path", type=str, default=None, help="Path to (custom) pretrained model path to train on. Default is none, so base model from DeciAI is used.")

    parser.add_argument("-d", "--out-dir", dest="out_dir", type=str, default="yolonas_trainings", help="Directory to save model file")
    parser.add_argument("-n", "--name", type=str, default="yolonas_train", help="Model name to save")
    
    return parser.parse_args()


def main():
    args = parse_args()

    train = train_yolo_nas(model_arch=args.model_arch,
                           batch=args.batch,
                           epochs=args.epochs,
                           workers=args.workers,
                           multi_gpu=args.multi_gpu,
                           num_gpus=args.num_gpus,
                           pretrained_checkpoint_path=args.model_path,
                           checkpoint_out_dir=args.out_dir,
                           experiment_name=args.name,
                           dataset_dir=args.dataset_dir)

    print("Finish")



if __name__ == "__main__":
    main()

