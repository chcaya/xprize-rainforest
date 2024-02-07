import os
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
from rastervision.core.data import make_od_scene, ClassConfig
from rastervision.pytorch_learner.dataset import ObjectDetectionSlidingWindowGeoDataset, \
    ObjectDetectionRandomWindowGeoDataset
from rastervision.pytorch_learner.dataset.visualizer import ObjectDetectionVisualizer
from rastervision.pytorch_learner import ObjectDetectionLearnerConfig, ObjectDetectionGeoDataConfig, SolverConfig
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from engine.detector.utils import display_train_valid_test_aoi


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

    # def forward(self, x, targets=None):
    #     if self.training and targets is None:
    #         raise ValueError("In training mode, targets must be passed to the forward method.")
    #     print("HERE")
    #     return self.model(x, targets)

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

        # visualize_image_with_boxes(images[0], targets[0], predictions[0])


class DetectorTrainingPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.working_folder = str(os.path.join(config["OUTPUT_FOLDER"], config["OUTPUT_NAME"]))

        if not os.path.exists(self.working_folder):
            os.mkdir(self.working_folder)

        self.rgb_path = str(os.path.join(config["DATA_ROOT"], config["TIF_NAME"]))
        self.trees_bbox_path = str(os.path.join(config["DATA_ROOT"], config["BBOX_TREES_NAME"]))
        self.train_bbox_path = str(os.path.join(config["DATA_ROOT"], config["BBOX_TRAIN_NAME"]))
        self.valid_bbox_path = str(os.path.join(config["DATA_ROOT"], config["BBOX_VALID_NAME"]))
        self.test_bbox_path = str(os.path.join(config["DATA_ROOT"], config["BBOX_TEST_NAME"]))

        self.chip_size = config["CHIP_SIZE"]
        self.chip_stride = config["CHIP_STRIDE"]

        self.class_config = ClassConfig(names=['background', 'tree'], colors=['white', 'green'])

        self.train_scene = self._create_scene(aoi_uri=self.train_bbox_path)
        self.valid_scene = self._create_scene(aoi_uri=self.valid_bbox_path)
        self.test_scene = self._create_scene(aoi_uri=self.test_bbox_path)

        self.train_dataset = ObjectDetectionRandomWindowGeoDataset(
            scene=self.train_scene,
            # window sizes will randomly vary from s1 to s2
            size_lims=(int(self.chip_size - 0.2 * self.chip_size), int(self.chip_size + 0.2 * self.chip_size)),
            # resize chips to nxn before returning
            out_size=self.chip_size,
            # allow windows to overflow the extent by n pixels
            padding=100,
            clip=True,
            normalize=True,
            ioa_thresh=0.9,
            neg_ioa_thresh=0.2,
            transform=self._get_data_augmentation_transform(),
            max_windows=500000,
            to_pytorch=True
        )

        self.valid_dataset = ObjectDetectionSlidingWindowGeoDataset(
            scene=self.valid_scene,
            size=self.chip_size,
            stride=self.chip_stride,
            normalize=True,
            to_pytorch=False
        )

        # for x, metadata in self.valid_dataset:
        #     print(x)
        #     print(metadata.)

        print(len([x for x in self.valid_dataset]))
        print(len(self.valid_dataset.windows))

        self.test_dataset = ObjectDetectionSlidingWindowGeoDataset(
            scene=self.test_scene,
            size=self.chip_size,
            stride=self.chip_stride,
            normalize=True
        )

        print(len([x for x in self.test_dataset]))
        print(len(self.test_dataset.windows))
        print(self.test_dataset.windows)

    def _create_scene(self, aoi_uri: str):
        scene = make_od_scene(
            class_config=self.class_config,
            image_uri=self.rgb_path,
            aoi_uri=aoi_uri,
            label_vector_uri=self.trees_bbox_path,
            label_vector_default_class_id=self.class_config.get_class_id('tree'),
            image_raster_source_kw=dict(allow_streaming=True))

        return scene

    def _get_data_augmentation_transform(self):
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
            ]),
        ])
        return data_augmentation_transform

    def plot_data_to_disk(self, show_images: bool):
        display_train_valid_test_aoi(train_scene=self.train_scene,
                                     valid_scene=self.valid_scene,
                                     test_scene=self.test_scene,
                                     show_image=show_images,
                                     output_file=str(os.path.join(self.working_folder, "detector_aois.png")))

        vis = ObjectDetectionVisualizer(
            class_names=self.class_config.names, class_colors=self.class_config.colors)

        x, y = vis.get_batch(self.train_dataset, 10)
        vis.plot_batch(x, y, output_path=str(os.path.join(self.working_folder, "detector_sample_tiles.png")),
                       show=show_images)

    def train(self, data_n_workers: int, output_name: str, batch_size: int, lr: float, n_epochs: int,
              backbone_resnet_out_channels: int):

        def custom_collate_fn(batch):
            images = []
            boxes = []
            for x in batch:
                # Convert image to tensor to get its shape if it's not already a tensor
                # This is necessary to clip the bounding boxes to the image size
                image_tensor = x[0]
                images.append(image_tensor)

                # Get image dimensions
                _, height, width = image_tensor.shape

                # Extract bounding boxes and labels
                raw_boxes = x[1].get_field('boxes')
                labels = x[1].get_field('class_ids')

                # Ensure all bounding box coordinates are within the image bounds
                # and no negative values. The bounding box format is assumed to be [x1, y1, x2, y2]
                clipped_boxes = torch.clamp(raw_boxes, min=0)
                clipped_boxes[:, 2] = torch.clamp(clipped_boxes[:, 2], max=width)
                clipped_boxes[:, 3] = torch.clamp(clipped_boxes[:, 3], max=height)
                boxes.append({
                    'boxes': clipped_boxes,
                    'labels': labels
                })

            return images, boxes

        train_dl = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                              collate_fn=custom_collate_fn)
        valid_dl = DataLoader(self.valid_dataset, batch_size=batch_size, num_workers=4,
                              collate_fn=custom_collate_fn)

        tensorboard_logger = TensorBoardLogger(save_dir=self.config["OUTPUT_NAME"], flush_secs=10)
        trainer = pl.Trainer(max_epochs=10,
                             logger=[tensorboard_logger])
        model = CustomFasterRCNN(lr=lr)

        # Train the model
        trainer.fit(model, train_dl, valid_dl)
