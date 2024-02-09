from typing import List

from config.config_parser.base_config_parser import BaseConfig
from dataclasses import dataclass


@dataclass
class PreprocessorConfig(BaseConfig):
    data_path: str
    annot_path: str
    output_folder: str
    tile_size: int
    tile_overlap: float
    scale_factor: int

    aoi_xmin: float
    aoi_ymin: float
    aoi_xmax: float
    aoi_ymax: float

    def __init__(self, config_path: str):
        super().__init__()
        config = self.load_yaml_config(config_path)

        self.data_path = config['data_path']
        self.annot_path = config['annot_path']
        self.output_folder = config['output_folder']
        self.tile_size = config['tile_size']
        self.tile_overlap = config['tile_overlap']
        self.scale_factor = config['scale_factor']
        self.aoi_xmin = config['area_of_interest']['xmin']
        self.aoi_ymin = config['area_of_interest']['ymin']
        self.aoi_xmax = config['area_of_interest']['xmax']
        self.aoi_ymax = config['area_of_interest']['ymax']


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
