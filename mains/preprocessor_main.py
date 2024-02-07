from dataclasses import dataclass
from pathlib import Path
from shapely import box

from config.base.base_config import BaseConfig
from engine.preprocessor.tilerizer import Tilerizer


@dataclass
class PreprocessorConfig(BaseConfig):
    data_path: str
    annot_path: str
    output_folder: str
    tile_size: int
    tile_overlap: float

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
        self.aoi_xmin = config['area_of_interest']['xmin']
        self.aoi_ymin = config['area_of_interest']['ymin']
        self.aoi_xmax = config['area_of_interest']['xmax']
        self.aoi_ymax = config['area_of_interest']['ymax']


def preprocessor_main(config_path: str):
    config = PreprocessorConfig(config_path)

    tilerizer = Tilerizer(data_path=Path(config.data_path),
                          annot_path=Path(config.annot_path))
    tilerizer.create_tiles(
        output_folder=Path(config.output_folder),
        tile_size=config.tile_size,
        tile_overlap=config.tile_overlap,
        area_of_interest=box(config.aoi_xmin, config.aoi_ymin, config.aoi_xmax, config.aoi_ymax))
