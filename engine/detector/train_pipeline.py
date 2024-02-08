import os
from pathlib import Path
from typing import Any

import torch
import torchmetrics.detection
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import albumentations as A

from config.config_parser.config_parsers import DetectorTrainingConfig
from engine.detector.utils import custom_collate_fn
from engine.detector.dataset import TilesObjectDetectionDataset


class CustomFasterRCNN(pl.LightningModule):
    def __init__(self, lr: float):
        super(CustomFasterRCNN, self).__init__()
        self.lr = lr

        # Load a pre-trained ResNet50 model
        backbone = resnet_fpn_backbone('resnet50', pretrained=True)

        # Define an anchor generator
        rpn_anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,), (512,)),  # Adjust the sizes as needed
            aspect_ratios=((0.5, 1.0, 2.0),) * 5  # Adjust the aspect ratios as needed
        )

        # Create the Faster R-CNN model
        self.model = FasterRCNN(
            backbone,
            num_classes=2,  # 1 class + background
            rpn_anchor_generator=rpn_anchor_generator,
            box_detections_per_img=100
        )

        self.map_metric = torchmetrics.detection.MeanAveragePrecision(iou_type="bbox",
                                                                      iou_thresholds=None,#[0.2, 0.4, 0.6],
                                                                      extended_summary=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, images, targets):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        predictions = self(images, targets)
        map_score = self.map_metric(predictions, targets)
        self.log_dict({'map': map_score['map']}, on_step=True, on_epoch=True, prog_bar=True, logger=True)


class DetectorTrainingPipeline:
    def __init__(self, config: DetectorTrainingConfig):
        self.config = config

    @staticmethod
    def _get_data_augmentation_transform():
        data_augmentation_transform = A.Compose([
                A.Flip(),
                A.ShiftScaleRotate(),
                A.OneOf([
                    A.HueSaturationValue(hue_shift_limit=10),
                    A.RGBShift(),
                    A.ToGray(),
                    A.ToSepia(),
                    A.RandomBrightness(),
                    A.RandomGamma(),
                ])
            ],
            bbox_params=A.BboxParams(
                format='pascal_voc',  # Specify the format of your bounding boxes
                label_fields=['labels'],  # Specify the field that contains the labels
                min_area=0.,  # Minimum area of a bounding box. All bboxes that have an area smaller than this value will be removed
                min_visibility=0.,  # Minimum visibility of a bounding box. All bboxes that have a visibility smaller than this value will be removed
            ))
        return data_augmentation_transform

    def train(self):
        train_ds = TilesObjectDetectionDataset(datasets_configs=self.config.train_datasets,
                                               transform=self._get_data_augmentation_transform())
        valid_ds = TilesObjectDetectionDataset(datasets_configs=self.config.valid_datasets,
                                               transform=self._get_data_augmentation_transform())

        train_dl = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True, num_workers=1,
                              collate_fn=custom_collate_fn)
        valid_dl = DataLoader(valid_ds, batch_size=self.config.batch_size, num_workers=1,
                              collate_fn=custom_collate_fn)

        tensorboard_logger = TensorBoardLogger(save_dir=Path(self.config.output_folder) / self.config.output_name,
                                               flush_secs=10)
        trainer = pl.Trainer(max_epochs=self.config.n_epochs,
                             logger=[tensorboard_logger])
        model = CustomFasterRCNN(lr=self.config.learning_rate)

        # Train the model
        trainer.fit(model, train_dl, valid_dl)
