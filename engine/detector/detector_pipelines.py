import warnings
from abc import ABC
from pathlib import Path

import torch
import torch.optim as optim
import torchmetrics
from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader
import albumentations as A
from tqdm import tqdm

from config.config_parsers.detector_parsers import DetectorTrainIOConfig, DetectorScoreIOConfig, DetectorInferIOConfig
from engine.detector.model import Detector
from geodataset.dataset import DetectionLabeledRasterCocoDataset, UnlabeledRasterDataset

from engine.detector.utils import WarmupStepLR


class DetectorBasePipeline(ABC):
    def __init__(self,
                 batch_size: int,
                 architecture: str,
                 checkpoint_state_dict_path: str,
                 backbone_model_resnet_name: str,
                 backbone_model_pretrained: bool,
                 box_predictions_per_image: int):

        self.batch_size = batch_size

        self.checkpoint_state_dict_path = checkpoint_state_dict_path
        self.box_predictions_per_image = box_predictions_per_image

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # No need to load the pretrained backbone model if we are starting from a custom checkpoint
        backbone_model_pretrained = False if checkpoint_state_dict_path else backbone_model_pretrained

        self.model = Detector(architecture=architecture,
                              backbone_model_resnet_name=backbone_model_resnet_name,
                              backbone_model_pretrained=backbone_model_pretrained,
                              box_predictions_per_image=box_predictions_per_image).to(self.device)

        if checkpoint_state_dict_path:
            self.model.load_state_dict(torch.load(checkpoint_state_dict_path))

        self.model.to(self.device)


class DetectorScorePipeline(DetectorBasePipeline):
    def __init__(self,
                 batch_size: int,
                 architecture: str,
                 checkpoint_state_dict_path: str,
                 backbone_model_resnet_name: str,
                 backbone_model_pretrained: bool,
                 box_predictions_per_image: int):

        super().__init__(batch_size=batch_size,
                         architecture=architecture,
                         checkpoint_state_dict_path=checkpoint_state_dict_path,
                         backbone_model_resnet_name=backbone_model_resnet_name,
                         backbone_model_pretrained=backbone_model_pretrained,
                         box_predictions_per_image=box_predictions_per_image)

        self.map_metric = torchmetrics.detection.MeanAveragePrecision(
            # backend='faster_coco_eval',   # Requires additional dependencies
            iou_type="bbox",
            # max_detection_thresholds=[1, 10, self.box_predictions_per_image]      # Causes issues for some reason ('map'=-1). Have to stick with the warning "Encountered more than 100 detections in a single image...".
        ).to(self.device)

    @classmethod
    def from_config(cls, detector_score_config: DetectorScoreIOConfig):
        return cls(batch_size=detector_score_config.base_params_config.batch_size,
                   architecture=detector_score_config.architecture_config.architecture_name,
                   checkpoint_state_dict_path=detector_score_config.checkpoint_state_dict_path,
                   backbone_model_resnet_name=detector_score_config.architecture_config.backbone_model_resnet_name,
                   backbone_model_pretrained=False,
                   box_predictions_per_image=detector_score_config.base_params_config.box_predictions_per_image)

    def _evaluate(self, data_loader, epoch=None):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            data_loader_with_progress = tqdm(data_loader,
                                             desc=f"Epoch {epoch + 1} (scoring)" if epoch is not None else "Scoring",
                                             leave=True)
            for batch_idx, (images, targets) in enumerate(data_loader_with_progress):
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                outputs = self.model(images, targets)
                predictions.extend(outputs)

                # Update MAP metric for validation
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=f".*{'Encountered more than 100 detections in a single image'}.*")
                    self.map_metric.update(outputs, targets)

        # Compute and log MAP metric
        scores = self.map_metric.compute()
        self.map_metric.reset()  # Reset metric for next epoch/validation

        return scores, predictions

    def score(self, test_ds: DetectionLabeledRasterCocoDataset, collate_fn: callable):
        test_dl = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False,
                             collate_fn=collate_fn,
                             num_workers=3, persistent_workers=True)

        scores, predictions = self._evaluate(test_dl)
        print(f"Score results: {scores}")
        return scores, predictions


class DetectorTrainPipeline(DetectorScorePipeline):
    def __init__(self, detector_train_config: DetectorTrainIOConfig):
        self.config = detector_train_config
        super().__init__(batch_size=self.config.base_params_config.batch_size,
                         architecture=self.config.architecture_config.architecture_name,
                         checkpoint_state_dict_path=self.config.start_checkpoint_state_dict_path,
                         backbone_model_resnet_name=self.config.architecture_config.backbone_model_resnet_name,
                         backbone_model_pretrained=self.config.backbone_model_pretrained,
                         box_predictions_per_image=self.config.base_params_config.box_predictions_per_image)

        self.output_folder = Path(self.config.output_folder)
        self.output_folder.mkdir(exist_ok=True, parents=True)

        self.model_output_folder = self.output_folder / self.config.output_name
        self.logs_folder = self.model_output_folder / "logs"
        self.config_folder = self.model_output_folder / "config"
        if self.model_output_folder.exists():
            raise Exception(f"The model output path {self.model_output_folder} already exists."
                            f" Please specify a new 'output_name'.")
        else:
            self.model_output_folder.mkdir(exist_ok=False)
            self.logs_folder.mkdir(exist_ok=False)
            self.config_folder.mkdir(exist_ok=False)

        self.config.save_yaml_config(self.config_folder / f"{self.config.output_name}_config.yaml")

        self.writer = SummaryWriter(log_dir=str(self.logs_folder),
                                    flush_secs=10)

    @classmethod
    def from_config(cls, detector_train_config: DetectorTrainIOConfig):
        return cls(detector_train_config=detector_train_config)

    def _save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def _train_one_epoch(self, optimizer, data_loader, epoch):
        self.model.train()
        accumulated_loss = 0.0
        data_loader_with_progress = tqdm(data_loader, desc=f"Epoch {epoch + 1} (training)", leave=True)
        for batch_idx, (images, targets) in enumerate(data_loader_with_progress):
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss = loss / self.config.grad_accumulation_steps

            loss.backward()
            accumulated_loss += loss.detach().item()

            # Perform optimization step only after accumulating enough gradients
            if (batch_idx + 1) % self.config.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Logging at intervals of train_log_interval effective batches
            if (batch_idx + 1) % (self.config.train_log_interval * self.config.grad_accumulation_steps) == 0:
                average_loss = accumulated_loss / self.config.train_log_interval
                self.writer.add_scalar('Loss/train', average_loss, self.config.base_params_config.batch_size * (epoch * len(data_loader) + batch_idx))
                accumulated_loss = 0.0

        # Ensure any remaining gradients are applied
        if (batch_idx + 1) % self.config.grad_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

    def train(self,
              train_ds: DetectionLabeledRasterCocoDataset,
              valid_ds: DetectionLabeledRasterCocoDataset,
              collate_fn: callable):
        params = [p for p in self.model.parameters() if p.requires_grad]

        optimizer = optim.SGD(
            params,
            lr=self.config.learning_rate,
            momentum=0.9,
            weight_decay=0.0005
        )

        scheduler = WarmupStepLR(
            optimizer,
            step_size=self.config.scheduler_step_size,
            gamma=self.config.scheduler_gamma,
            warmup_steps=self.config.scheduler_warmup_steps,
            base_lr=self.config.learning_rate / 100
        )

        train_dl = DataLoader(train_ds, batch_size=self.config.base_params_config.batch_size, shuffle=True,
                              collate_fn=collate_fn,
                              num_workers=3, persistent_workers=True)
        valid_dl = DataLoader(valid_ds, batch_size=self.config.base_params_config.batch_size, shuffle=False,
                              collate_fn=collate_fn,
                              num_workers=3, persistent_workers=True)

        print(f"Training for {self.config.n_epochs} epochs...")
        print(f'Effective batch size'
              f' = batch_size * grad_accumulation_steps'
              f' = {self.config.base_params_config.batch_size} * {self.config.grad_accumulation_steps}'
              f' = {self.config.base_params_config.batch_size * self.config.grad_accumulation_steps}')

        for epoch in range(self.config.n_epochs):
            # also log the current learning rate
            self.writer.add_scalar('lr', scheduler.get_lr()[0], epoch)

            self._train_one_epoch(optimizer, train_dl, epoch=epoch)
            scores, predictions = self._evaluate(valid_dl, epoch=epoch)
            self.writer.add_scalar('metric/map', scores['map'], epoch)
            self.writer.add_scalar('metric/map_50', scores['map_50'], epoch)
            self.writer.add_scalar('metric/map_75', scores['map_75'], epoch)

            scheduler.step()

            if epoch % self.config.save_model_every_n_epoch == 0:
                self._save_model(save_path=self.model_output_folder / f"{self.config.output_name}_{epoch}.pt")

    @staticmethod
    def get_data_augmentation_transform():
        data_augmentation_transform = A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            # A.ShiftScaleRotate(p=0.2),        # this can put boxes out of the image and then the training crashes on an image without boxes

            A.RandomBrightnessContrast(p=0.1),
            A.HueSaturationValue(hue_shift_limit=10, p=0.1),
            A.RGBShift(p=0.1),
            A.RandomGamma(p=0.1),
            A.Blur(p=0.1),
            A.ToGray(p=0.02),
            A.ToSepia(p=0.02),
        ],
            bbox_params=A.BboxParams(
                format='pascal_voc',  # Specify the format of your bounding boxes
                label_fields=['labels'],  # Specify the field that contains the labels
                min_area=0.,
                # Minimum area of a bounding box. All bboxes that have an area smaller than this value will be removed
                min_visibility=0.,
                # Minimum visibility of a bounding box. All bboxes that have a visibility smaller than this value will be removed
            ))
        return data_augmentation_transform


class DetectorInferencePipeline(DetectorBasePipeline):
    def __init__(self,
                 batch_size: int,
                 architecture: str,
                 checkpoint_state_dict_path: str,
                 backbone_model_resnet_name: str,
                 backbone_model_pretrained: bool,
                 box_predictions_per_image: int):

        super().__init__(batch_size=batch_size,
                         architecture=architecture,
                         checkpoint_state_dict_path=checkpoint_state_dict_path,
                         backbone_model_resnet_name=backbone_model_resnet_name,
                         backbone_model_pretrained=backbone_model_pretrained,
                         box_predictions_per_image=box_predictions_per_image)

    @classmethod
    def from_config(cls, config: DetectorInferIOConfig):
        return cls(batch_size=config.base_params_config.batch_size,
                   architecture=config.architecture_config.architecture_name,
                   checkpoint_state_dict_path=config.checkpoint_state_dict_path,
                   backbone_model_resnet_name=config.architecture_config.backbone_model_resnet_name,
                   backbone_model_pretrained=False,
                   box_predictions_per_image=config.base_params_config.box_predictions_per_image)

    def _infer(self, data_loader):
        self.model.eval()

        predictions = []

        with torch.no_grad():
            data_loader_with_progress = tqdm(data_loader,
                                             desc="Inferring...",
                                             leave=True)
            for images in data_loader_with_progress:
                images = list(img.to(self.device) for img in images)
                outputs = self.model(images)
                predictions.extend(outputs)

        return predictions

    def infer(self, infer_ds: UnlabeledRasterDataset, collate_fn: callable):
        infer_dl = DataLoader(infer_ds, batch_size=self.batch_size, shuffle=False,
                              collate_fn=collate_fn,
                              num_workers=3, persistent_workers=True)

        results = self._infer(infer_dl)
        return results

