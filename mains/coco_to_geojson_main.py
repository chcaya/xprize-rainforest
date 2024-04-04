from pathlib import Path

from geodataset.utils import CocoNameConvention, GeoJsonNameConvention, coco_to_geojson


from config.config_parsers.coco_to_geojson_parsers import CocoToGeojsonIOConfig


def coco_to_geojson_main(config: CocoToGeojsonIOConfig):
    output_folder = Path(config.output_folder)
    output_folder.mkdir(exist_ok=False, parents=True)

    product_name, scale_factor, ground_resolution, fold = CocoNameConvention.parse_name(Path(config.coco_path).name)

    geojson_name = GeoJsonNameConvention.create_name(product_name=product_name,
                                                     fold=fold,
                                                     scale_factor=scale_factor,
                                                     ground_resolution=ground_resolution)

    gdf = coco_to_geojson(
        coco_json_path=config.coco_path,
        images_directory=config.input_tiles_root,
        geojson_output_path=f"{config.output_folder}/{geojson_name}"
    )

    return gdf
