from typing import List

from config.config_parser.base_config_parser import BaseConfig
from dataclasses import dataclass


@dataclass
class PreprocessorConfig(BaseConfig):
    raster_path: str
    labels_path: str
    output_folder: str
    tile_size: int
    tile_overlap: float
    scale_factor: int

    aoi_config: str
    aoi_type: str
    aois: dict

    ignore_black_white_alpha_tiles_threshold: float
    ignore_tiles_without_labels: bool
    main_label_category_column_name: str

    def __init__(self, config_path: str):
        super().__init__()
        config = self.load_yaml_config(config_path)

        self.raster_path = config['raster_path']
        self.labels_path = config['labels_path']
        self.output_folder = config['output_folder']
        self.tile_size = config['tile_size']
        self.tile_overlap = config['tile_overlap']
        self.scale_factor = config['scale_factor']
        self.aoi_config = config['area_of_interest']['aoi_config']
        self.aoi_type = config['area_of_interest']['aoi_type']
        self.aois = config['area_of_interest']['aois']
        self.ignore_black_white_alpha_tiles_threshold = config['ignore_black_white_alpha_tiles_threshold']
        self.ignore_tiles_without_labels = config['ignore_tiles_without_labels']
        self.main_label_category_column_name = config['main_label_category_column_name']


@dataclass
class DetectorTrainingConfig(BaseConfig):
    output_folder: str
    output_name: str

    train_datasets: List[dict]
    valid_datasets: List[dict]
    test_datasets: List[dict]

    chip_size: int
    chip_stride: int

    batch_size: int
    learning_rate: float
    scheduler_step_size: int
    scheduler_gamma: float
    n_epochs: int
    rcnn_backbone_model_resnet_name: str
    rcnn_backbone_model_pretrained: bool
    backbone_resnet_out_channels: int
    box_predictions_per_image: int

    def __init__(self, config_path: str):
        super().__init__()
        config = self.load_yaml_config(config_path)

        self.output_folder = config['output']['output_folder']
        self.output_name = config['output']['output_name']

        self.train_datasets = config['datasets']['train']
        self.valid_datasets = config['datasets']['valid']
        self.test_datasets = config['datasets']['test']

        self.chip_size = config['tiles']['chip_size']
        self.chip_stride = config['tiles']['chip_stride']

        self.batch_size = config['model_config']['batch_size']
        self.learning_rate = float(config['model_config']['learning_rate'])
        self.scheduler_step_size = config['model_config']['scheduler_step_size']
        self.scheduler_gamma = config['model_config']['scheduler_gamma']
        self.n_epochs = config['model_config']['n_epochs']
        self.rcnn_backbone_model_resnet_name = config['model_config']['rcnn_backbone_model_resnet_name']
        self.rcnn_backbone_model_pretrained = config['model_config']['rcnn_backbone_model_pretrained']
        self.backbone_resnet_out_channels = config['model_config']['backbone_resnet_out_channels']
        self.box_predictions_per_image = config['model_config']['box_predictions_per_image']
