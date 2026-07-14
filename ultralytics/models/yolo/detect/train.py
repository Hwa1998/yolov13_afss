# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
import random
from copy import copy

import numpy as np
import torch
import torch.nn as nn

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK, TQDM
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import non_max_suppression
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first


class DetectionTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionTrainer

        args = dict(model="yolo11n.pt", data="coco8.yaml", epochs=3)
        trainer = DetectionTrainer(overrides=args)
        trainer.train()
        ```
    """

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Construct and return dataloader."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """Get batch size by calculating memory occupation of model."""
        train_dataset = self.build_dataset(self.trainset, mode="train", batch=16)
        # 4 for mosaic augmentation
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4
        return super().auto_batch(max_num_obj)

    def compute_per_image_metrics(self, iou_thresh=0.5, conf_thresh=0.25):
        """
        Compute per-image precision and recall on the training set.

        Runs a lightweight inference pass (no augmentation, no backprop) using
        the EMA model to evaluate detection quality for each training image.
        Returns arrays of per-image precision and recall values.

        Args:
            iou_thresh: IoU threshold for matching predictions to GT (default 0.5).
            conf_thresh: Confidence threshold for NMS (default 0.25).

        Returns:
            Tuple of (image_indices, precisions, recalls) as numpy arrays.
        """
        # Use EMA model for more stable predictions, fall back to main model
        model = self.ema.ema if self.ema else self.model
        model.eval()

        # Use cached eval dataset and loader to avoid rebuilding every AFSS update
        if not hasattr(self, '_afss_eval_loader') or self._afss_eval_loader is None:
            gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
            self._afss_eval_dataset = build_yolo_dataset(
                self.args, self.trainset, self.batch_size, self.data,
                mode="val", rect=False, stride=gs,
            )
            # Use fewer workers for eval to reduce spawn/kill overhead on Windows
            eval_workers = max(0, min(self.args.workers, 4))
            self._afss_eval_loader = build_dataloader(
                self._afss_eval_dataset, self.batch_size * 2,
                eval_workers, shuffle=False, rank=-1,
            )
        eval_loader = self._afss_eval_loader

        n_images = len(self._afss_eval_dataset)
        precisions = np.zeros(n_images, dtype=np.float32)
        recalls = np.zeros(n_images, dtype=np.float32)

        seen = 0
        with torch.no_grad():
            pbar = TQDM(eval_loader, desc="AFSS: Computing per-image metrics")
            for batch in pbar:
                batch = self.preprocess_batch(batch)
                preds = model(batch["img"])

                # Parse model output based on head type
                preds_nms = None  # will be set below

                if isinstance(preds, dict):
                    # Detect_NMSFree: dict where "one2one" = (decoded_tensor, raw_features)
                    # decoded_tensor shape: (batch, 4*reg_max + nc, num_anchors) in raw DFL format
                    one2one = preds["one2one"]
                    if isinstance(one2one, (tuple, list)):
                        one2one = one2one[0]  # extract the decoded tensor
                    # Use the head's decode_bboxes to get xywh boxes (applies DFL + anchor decode)
                    model_head = de_parallel(self.model).model[-1]
                    box_raw, cls_raw = one2one.split((model_head.reg_max * 4, model_head.nc), 1)
                    dbox = model_head.decode_bboxes(box_raw)  # xywh pixel coords, shape (batch, 4, A)
                    # Combine decoded boxes with class logits (sigmoid applied by NMS)
                    decoded_preds = torch.cat((dbox, cls_raw.sigmoid()), 1)  # (batch, 4+nc, A)
                    # Apply standard NMS to get xyxy boxes
                    preds_nms = non_max_suppression(
                        decoded_preds,
                        conf_thresh,
                        iou_thres=0.5,
                        multi_label=True,
                        agnostic=self.args.single_cls,
                        max_det=self.args.max_det,
                    )
                else:
                    if isinstance(preds, (list, tuple)):
                        preds = preds[0]
                    # Standard Detect: apply NMS, output [x1,y1,x2,y2,conf,cls]
                    preds_nms = non_max_suppression(
                        preds,
                        conf_thresh,
                        iou_thres=0.5,
                        multi_label=True,
                        agnostic=self.args.single_cls,
                        max_det=self.args.max_det,
                    )

                # Process each image in the batch
                for si, pred in enumerate(preds_nms):
                    img_idx = seen + si
                    if img_idx >= n_images:
                        break

                    # Get GT for this image (move to GPU to match pred_boxes device)
                    device = batch["img"].device
                    batch_idx_mask = batch["batch_idx"] == si
                    gt_cls = batch["cls"][batch_idx_mask].squeeze(-1).to(device)
                    gt_bboxes = batch["bboxes"][batch_idx_mask].to(device)

                    n_gt = len(gt_cls)
                    n_pred = len(pred)

                    if n_gt == 0 and n_pred == 0:
                        # No objects, no predictions -> perfect
                        precisions[img_idx] = 1.0
                        recalls[img_idx] = 1.0
                    elif n_gt == 0:
                        # No GT but predictions exist -> precision=0, recall=1 (vacuously)
                        precisions[img_idx] = 0.0
                        recalls[img_idx] = 1.0
                    elif n_pred == 0:
                        # GT exists but no predictions -> precision=1 (vacuously), recall=0
                        precisions[img_idx] = 1.0
                        recalls[img_idx] = 0.0
                    else:
                        # Convert both to xyxy for IoU
                        pred_boxes = pred[:, :4]  # already xyxy from NMS
                        # GT boxes are in normalized xywh, convert to pixel xyxy
                        img_h, img_w = batch["img"].shape[2:]
                        gt_xyxy = gt_bboxes.clone()
                        # xywh -> xyxy
                        gt_xyxy[:, 0] = (gt_bboxes[:, 0] - gt_bboxes[:, 2] / 2) * img_w
                        gt_xyxy[:, 1] = (gt_bboxes[:, 1] - gt_bboxes[:, 3] / 2) * img_h
                        gt_xyxy[:, 2] = (gt_bboxes[:, 0] + gt_bboxes[:, 2] / 2) * img_w
                        gt_xyxy[:, 3] = (gt_bboxes[:, 1] + gt_bboxes[:, 3] / 2) * img_h

                        # Compute IoU matrix (n_gt x n_pred)
                        iou = box_iou(gt_xyxy, pred_boxes)

                        # Match predictions to GT: for each GT, find best matching pred
                        matched_gt = set()
                        matched_pred = set()
                        tp = 0

                        # Greedy matching by highest IoU
                        while True:
                            if iou.numel() == 0:
                                break
                            max_iou, max_idx = iou.flatten().max(0)
                            if max_iou < iou_thresh:
                                break
                            gi, pi = max_idx.item() // iou.shape[1], max_idx.item() % iou.shape[1]
                            if gi in matched_gt or pi in matched_pred:
                                iou[gi, pi] = 0  # zero out and try next
                                continue
                            matched_gt.add(gi)
                            matched_pred.add(pi)
                            # Also check class match
                            if int(gt_cls[gi]) == int(pred[pi, 5]):
                                tp += 1
                            iou[gi, pi] = 0  # prevent re-matching

                        fp = n_pred - tp
                        fn = n_gt - tp

                        precisions[img_idx] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        recalls[img_idx] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

                seen += len(preds_nms)

        model.train()
        image_indices = np.arange(n_images)
        return image_indices, precisions, recalls
