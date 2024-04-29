from pathlib import Path


from geodataset.tilerize import RasterTilerizer, LabeledRasterTilerizer, PolygonTilerizer

from config.config_parsers.tilerizer_parsers import TilerizerIOConfig
from engine.tilerizer.utils import parse_tilerizer_aoi_config


def tilerizer_main(config: TilerizerIOConfig):
    output_folder = Path(config.output_folder)
    output_folder.mkdir(exist_ok=False, parents=True)

    aois_config = parse_tilerizer_aoi_config(config)
    if config.tile_type == 'tile':
        if config.labels_path:
            tilerizer = LabeledRasterTilerizer(
                raster_path=Path(config.raster_path),
                labels_path=Path(config.labels_path),
                output_path=Path(config.output_folder),
                tile_size=config.tile_size,
                tile_overlap=config.tile_overlap,
                aois_config=aois_config,
                scale_factor=config.raster_resolution_config.scale_factor,
                ground_resolution=config.raster_resolution_config.ground_resolution,
                ignore_black_white_alpha_tiles_threshold=config.ignore_black_white_alpha_tiles_threshold,
                ignore_tiles_without_labels=config.ignore_tiles_without_labels,
                main_label_category_column_name=config.main_label_category_column_name)

            coco_paths = tilerizer.generate_coco_dataset()
        else:
            tilerizer = RasterTilerizer(
                raster_path=Path(config.raster_path),
                output_path=Path(config.output_folder),
                tile_size=config.tile_size,
                tile_overlap=config.tile_overlap,
                aois_config=aois_config,
                scale_factor=config.raster_resolution_config.scale_factor,
                ground_resolution=config.raster_resolution_config.ground_resolution,
                ignore_black_white_alpha_tiles_threshold=config.ignore_black_white_alpha_tiles_threshold,
            )

            tilerizer.generate_tiles()
            coco_paths = None
    elif config.tile_type == 'polygon':
        tilerizer = PolygonTilerizer(
            raster_path=Path(config.raster_path),
            output_path=Path(config.output_folder),
            labels_path=Path(config.labels_path),
            tile_size=config.tile_size,
            aois_config=aois_config,
            scale_factor=config.raster_resolution_config.scale_factor,
            ground_resolution=config.raster_resolution_config.ground_resolution,
        )
        coco_paths = tilerizer.generate_coco_dataset()
    else:
        raise ValueError(f"Invalid tile type: {config.tile_type}. Expected 'tile' or 'polygon'.")

    config.save_yaml_config(output_path=output_folder / "tilerizer_config.yaml")

    return tilerizer.tiles_path, coco_paths


