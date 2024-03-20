from pathlib import Path
from typing import Dict

import yaml

from config.config_parser.base_config_parser import BaseConfig
from dataclasses import dataclass


@dataclass
class PreprocessorConfig(BaseConfig):
    raster_path: str
    labels_path: str or None
    output_folder: str
    tile_size: int
    tile_overlap: float
    scale_factor: float
    ground_resolution: float

    aoi_config: str
    aoi_type: str
    aois: Dict

    ignore_black_white_alpha_tiles_threshold: float
    ignore_tiles_without_labels: bool
    main_label_category_column_name: str or None

    @classmethod
    def from_config_path(cls, config_path: str):
        config = cls.load_yaml_config(config_path)

        return cls(
            raster_path=config['raster_path'],
            labels_path=config['labels_path'],
            output_folder=config['output_folder'],
            tile_size=config['tile_size'],
            tile_overlap=config['tile_overlap'],
            scale_factor=config['scale_factor'],
            ground_resolution=config['ground_resolution'],
            aoi_config=config['area_of_interest']['aoi_config'],
            aoi_type=config['area_of_interest']['aoi_type'],
            aois=config['area_of_interest']['aois'],
            ignore_black_white_alpha_tiles_threshold=config['ignore_black_white_alpha_tiles_threshold'],
            ignore_tiles_without_labels=config['ignore_tiles_without_labels'],
            main_label_category_column_name=config['main_label_category_column_name'],
        )

    def save_yaml_config(self, output_path: Path):
        # Structure mirroring from_config_path
        config = {
            'raster_path': self.raster_path,
            'labels_path': self.labels_path,
            'output_folder': self.output_folder,
            'tile_size': self.tile_size,
            'tile_overlap': self.tile_overlap,
            'scale_factor': self.scale_factor,
            'ground_resolution': self.ground_resolution,
            'area_of_interest': {
                'aoi_config': self.aoi_config,
                'aoi_type': self.aoi_type,
                'aois': self.aois
            },
            'ignore_black_white_alpha_tiles_threshold': self.ignore_black_white_alpha_tiles_threshold,
            'ignore_tiles_without_labels': self.ignore_tiles_without_labels,
            'main_label_category_column_name': self.main_label_category_column_name,
        }

        with output_path.open('w') as file:
            yaml.dump(config, file, sort_keys=False)


@dataclass
class DetectorTrainConfig(BaseConfig):
    output_folder: str
    output_name: str
    train_log_interval: int

    data_root_path: str

    architecture: str
    rcnn_backbone_model_resnet_name: str
    rcnn_backbone_model_pretrained: bool

    start_checkpoint_state_dict_path: str
    batch_size: int
    learning_rate: float
    scheduler_step_size: int
    scheduler_gamma: float
    n_epochs: int
    save_model_every_n_epoch: int
    backbone_resnet_out_channels: int
    box_predictions_per_image: int

    @classmethod
    def from_config_path(cls, config_path: str):
        config = cls.load_yaml_config(config_path)
        train_config = config['train']
        architecture_config = config['architecture']
        model_config = train_config['model_config']

        return cls(
            output_folder=train_config['output']['output_folder'],
            output_name=train_config['output']['output_name'],
            train_log_interval=train_config['output']['train_log_interval'],
            data_root_path=config['data_root_path'],
            architecture=architecture_config['name'],
            rcnn_backbone_model_resnet_name=architecture_config['rcnn_backbone_model_resnet_name'],
            rcnn_backbone_model_pretrained=model_config['rcnn_backbone_model_pretrained'],
            start_checkpoint_state_dict_path=model_config['start_checkpoint_state_dict_path'],
            batch_size=config['batch_size'],
            learning_rate=model_config['learning_rate'],
            scheduler_step_size=model_config['scheduler_step_size'],
            scheduler_gamma=model_config['scheduler_gamma'],
            n_epochs=model_config['n_epochs'],
            save_model_every_n_epoch=model_config['save_model_every_n_epoch'],
            backbone_resnet_out_channels=model_config['backbone_resnet_out_channels'],
            box_predictions_per_image=model_config['box_predictions_per_image'],
        )

    def save_yaml_config(self, output_path: Path):
        config = {
            'data_root_path': self.data_root_path,
            'batch_size': self.batch_size,
            'train': {
                'output': {
                    'output_folder': self.output_folder,
                    'output_name': self.output_name,
                    'train_log_interval': self.train_log_interval
                },
                'model_config': {
                    'rcnn_backbone_model_pretrained': self.rcnn_backbone_model_pretrained,
                    'start_checkpoint_state_dict_path': self.start_checkpoint_state_dict_path,
                    'learning_rate': self.learning_rate,
                    'scheduler_step_size': self.scheduler_step_size,
                    'scheduler_gamma': self.scheduler_gamma,
                    'n_epochs': self.n_epochs,
                    'save_model_every_n_epoch': self.save_model_every_n_epoch,
                    'backbone_resnet_out_channels': self.backbone_resnet_out_channels,
                    'box_predictions_per_image': self.box_predictions_per_image
                }
            },
            'architecture': {
                'name': self.architecture,
                'rcnn_backbone_model_resnet_name': self.rcnn_backbone_model_resnet_name
            }
        }

        with output_path.open('w') as file:
            yaml.dump(config, file, sort_keys=False)


@dataclass
class DetectorScoreConfig(BaseConfig):
    data_root_path: str
    architecture: str
    rcnn_backbone_model_resnet_name: str
    batch_size: int
    checkpoint_state_dict_path: str
    box_predictions_per_image: int

    @classmethod
    def from_config_path(cls, config_path: str):
        config = cls.load_yaml_config(config_path)
        score_config = config['score']
        architecture_config = config['architecture']

        return cls(
            data_root_path=config['data_root_path'],
            architecture=architecture_config['name'],
            rcnn_backbone_model_resnet_name=architecture_config['rcnn_backbone_model_resnet_name'],
            batch_size=config['batch_size'],
            checkpoint_state_dict_path=score_config['checkpoint_state_dict_path'],
            box_predictions_per_image=score_config['box_predictions_per_image']
        )

    def save_yaml_config(self, output_path: Path):
        config = {
            'data_root_path': self.data_root_path,
            'batch_size': self.batch_size,
            'score': {
                'checkpoint_state_dict_path': self.checkpoint_state_dict_path,
                'box_predictions_per_image': self.box_predictions_per_image
            },
            'architecture': {
                'name': self.architecture,
                'rcnn_backbone_model_resnet_name': self.rcnn_backbone_model_resnet_name
            }
        }

        with output_path.open('w') as file:
            yaml.dump(config, file, sort_keys=False)


@dataclass
class DetectorInferConfig(DetectorScoreConfig):
    pass


@dataclass
class XPrizeConfig(BaseConfig):
    raster_path: str
    output_folder: str
    tile_size: int
    tile_overlap: float
    scale_factor: float
    ground_resolution: float

    aoi_config: str
    aoi_type: str
    aois: Dict

    ignore_black_white_alpha_tiles_threshold: float

    detector_batch_size: int
    detector_checkpoint_state_dict_path: str
    detector_architecture: str
    detector_rcnn_backbone_model_resnet_name: str
    detector_box_predictions_per_image: int

    aggregator_score_threshold: float
    aggregator_nms_threshold: float
    aggregator_nms_algorithm: str

    @classmethod
    def from_config_path(cls, config_path: str):
        config = cls.load_yaml_config(config_path)
        tilerizer_config = config['tilerizer']
        detector_config = config['detector']
        aggregator_config = config['aggregator']

        return cls(
            raster_path=config['raster_path'],
            output_folder=config['output_folder'],
            tile_size=tilerizer_config['tile_size'],
            tile_overlap=tilerizer_config['tile_overlap'],
            scale_factor=tilerizer_config['scale_factor'],
            ground_resolution=tilerizer_config['ground_resolution'],
            aoi_config=tilerizer_config['area_of_interest']['aoi_config'],
            aoi_type=tilerizer_config['area_of_interest']['aoi_type'],
            aois=tilerizer_config['area_of_interest']['aois'],
            ignore_black_white_alpha_tiles_threshold=tilerizer_config['ignore_black_white_alpha_tiles_threshold'],
            detector_batch_size=detector_config['batch_size'],
            detector_checkpoint_state_dict_path=detector_config['checkpoint_state_dict_path'],
            detector_architecture=detector_config['architecture']['name'],
            detector_rcnn_backbone_model_resnet_name=detector_config['architecture']['rcnn_backbone_model_resnet_name'],
            detector_box_predictions_per_image=detector_config['box_predictions_per_image'],
            aggregator_score_threshold=aggregator_config['score_threshold'],
            aggregator_nms_threshold=aggregator_config['nms_threshold'],
            aggregator_nms_algorithm=aggregator_config['nms_algorithm']
        )

    def save_yaml_config(self, output_path: Path):
        config = {
            'raster_path': self.raster_path,
            'output_folder': self.output_folder,
            'tilerizer': {
                'tile_size': self.tile_size,
                'tile_overlap': self.tile_overlap,
                'scale_factor': self.scale_factor,
                'ground_resolution': self.ground_resolution,
                'area_of_interest': {
                    'aoi_config': self.aoi_config,
                    'aoi_type': self.aoi_type,
                    'aois': self.aois
                },
                'ignore_black_white_alpha_tiles_threshold': self.ignore_black_white_alpha_tiles_threshold
            },
            'detector': {
                'batch_size': self.detector_batch_size,
                'checkpoint_state_dict_path': self.detector_checkpoint_state_dict_path,
                'architecture': {
                    'name': self.detector_architecture,
                    'rcnn_backbone_model_resnet_name': self.detector_rcnn_backbone_model_resnet_name
                },
                'box_predictions_per_image': self.detector_box_predictions_per_image
            }
        }

        with output_path.open('w') as file:
            yaml.dump(config, file, sort_keys=False)
