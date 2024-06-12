from abc import abstractmethod, ABC
from pathlib import Path

from geodataset.utils import strip_all_extensions
from geodataset.utils.file_name_conventions import validate_and_convert_product_name

from config.config_parsers.coco_to_geopackage_parsers import CocoToGeopackageIOConfig
from config.config_parsers.tilerizer_parsers import TilerizerIOConfig, TilerizerConfig


class BaseRasterPipeline(ABC):
    def __init__(self,
                 raster_path: str,
                 output_folder: str):
        self.raster_path = raster_path
        self.raster_name = validate_and_convert_product_name(strip_all_extensions(Path(self.raster_path)))
        self.output_folder = output_folder

    @abstractmethod
    def run(self):
        pass

    def _get_tilerizer_config(self,
                              tilerizer_config: TilerizerConfig,
                              output_folder: Path,
                              labels_path: Path or None,
                              main_label_category_column_name: str or None,
                              other_labels_attributes_column_names: list or None):
        assert len(tilerizer_config.aois) == 1, \
            "Only one AOI for the tilerizer is supported for now in the XPrize infer pipeline."

        preprocessor_config = TilerizerIOConfig(
            **tilerizer_config.as_dict(),
            raster_path=self.raster_path,
            output_folder=str(output_folder),
            labels_path=labels_path,
            ignore_tiles_without_labels=True,
            main_label_category_column_name=main_label_category_column_name,
            other_labels_attributes_column_names=other_labels_attributes_column_names
        )

        return preprocessor_config

    @staticmethod
    def _get_coco_to_geopackage_config(input_tiles_root: Path,
                                       coco_path: Path,
                                       output_folder: Path):
        coco_to_geopackage_config = CocoToGeopackageIOConfig(
            input_tiles_root=str(input_tiles_root),
            coco_path=str(coco_path),
            output_folder=str(output_folder),
        )

        return coco_to_geopackage_config
