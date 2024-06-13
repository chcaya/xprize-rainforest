from dataclasses import dataclass

from config.config_parsers.base_config_parsers import BaseConfig, BaseIntermediateConfig


@dataclass
class RasterResolutionConfig(BaseIntermediateConfig):
    scale_factor: float
    ground_resolution: float

    @classmethod
    def from_dict(cls, config: dict):
        return cls(**config)

    def to_structured_dict(self):
        config = {
            'scale_factor': self.scale_factor,
            'ground_resolution': self.ground_resolution,
        }

        return config


@dataclass
class TilerizerConfig(BaseConfig):
    tile_type: str
    tile_size: int
    tile_overlap: float
    raster_resolution_config: RasterResolutionConfig

    aoi_config: str
    aoi_type: str
    aois: dict

    ignore_black_white_alpha_tiles_threshold: float

    @classmethod
    def from_dict(cls, config: dict):
        tilerizer_config = config['tilerizer']
        aoi_config = tilerizer_config['area_of_interest']
        raster_resolution_config = RasterResolutionConfig.from_dict(tilerizer_config['raster_resolution'])

        return cls(
            tile_type=tilerizer_config['tile_type'],
            tile_size=tilerizer_config['tile_size'],
            tile_overlap=tilerizer_config['tile_overlap'],
            raster_resolution_config=raster_resolution_config,
            aoi_config=aoi_config['aoi_config'],
            aoi_type=aoi_config['aoi_type'],
            aois=aoi_config['aois'],
            ignore_black_white_alpha_tiles_threshold=tilerizer_config['ignore_black_white_alpha_tiles_threshold']
        )

    def to_structured_dict(self):
        config = {
            'tilerizer': {
                'tile_type': self.tile_type,
                'tile_size': self.tile_size,
                'tile_overlap': self.tile_overlap,
                'raster_resolution_config': self.raster_resolution_config.to_structured_dict(),
                'area_of_interest': {
                    'aoi_config': self.aoi_config,
                    'aoi_type': self.aoi_type,
                    'aois': self.aois
                },
                'ignore_black_white_alpha_tiles_threshold': self.ignore_black_white_alpha_tiles_threshold,
            }
        }

        return config


@dataclass
class TilerizerIOConfig(TilerizerConfig):
    raster_path: str
    output_folder: str
    labels_path: str or None
    ignore_tiles_without_labels: bool
    main_label_category_column_name: str or None
    other_labels_attributes_column_names: list or None

    @classmethod
    def from_dict(cls, config: dict):
        parent_config = TilerizerConfig.from_dict(config)
        tilerizer_config = config['tilerizer']
        tilerizer_io_config = tilerizer_config['io']

        return cls(
            **parent_config.as_dict(),
            raster_path=tilerizer_io_config['raster_path'],
            output_folder=tilerizer_io_config['output_folder'],
            labels_path=tilerizer_io_config['labels_path'],
            ignore_tiles_without_labels=tilerizer_io_config['ignore_tiles_without_labels'],
            main_label_category_column_name=tilerizer_io_config['main_label_category_column_name'],
            other_labels_attributes_column_names=tilerizer_io_config['other_labels_attributes_column_names']
        )

    def to_structured_dict(self):
        config = super().to_structured_dict()
        config['tilerizer']['io'] = {
            'raster_path': self.raster_path,
            'output_folder': self.output_folder,
            'labels_path': self.labels_path,
            'ignore_tiles_without_labels': self.ignore_tiles_without_labels,
            'main_label_category_column_name': self.main_label_category_column_name,
            'other_labels_attributes_column_names': self.other_labels_attributes_column_names
        }
