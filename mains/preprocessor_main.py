from pathlib import Path
from shapely import box

from config.config_parser.config_parsers import PreprocessorConfig
from engine.preprocessor.tilerizer import Tilerizer


def preprocessor_main(config_path: str):
    config = PreprocessorConfig(config_path)

    tilerizer = Tilerizer(data_path=Path(config.data_path),
                          annot_path=Path(config.annot_path))
    tilerizer.create_tiles(
        output_folder=Path(config.output_folder),
        tile_size=config.tile_size,
        tile_overlap=config.tile_overlap,
        area_of_interest=box(config.aoi_xmin, config.aoi_ymin, config.aoi_xmax, config.aoi_ymax))
