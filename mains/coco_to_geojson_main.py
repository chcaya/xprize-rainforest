from pathlib import Path

from geodataset.utils import CocoNameConvention, GeoJsonNameConvention, coco_to_geojson


from config.config_parsers.coco_to_geojson_parsers import CocoToGeojsonIOConfig


def coco_to_geojson_main(config: CocoToGeojsonIOConfig):
    output_folder = Path(config.output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    product_name, scale_factor, ground_resolution, fold = CocoNameConvention.parse_name(Path(config.coco_path).name)

    geojson_name = GeoJsonNameConvention.create_name(product_name=product_name,
                                                     fold=fold,
                                                     scale_factor=scale_factor,
                                                     ground_resolution=ground_resolution)

    geojson_output_path = output_folder / geojson_name

    gdf = coco_to_geojson(
        coco_json_path=config.coco_path,
        images_directory=config.input_tiles_root,
        convert_to_crs_coordinates=True,
        geojson_output_path=str(geojson_output_path),
    )

    return gdf, geojson_output_path
