from pathlib import Path

from geodataset.aggregator import DetectorAggregator, SegmentationAggregator
from geodataset.utils import CocoNameConvention

from config.config_parsers.aggregator_parsers import AggregatorIOConfig


def aggregator_main(config: AggregatorIOConfig):
    output_folder = Path(config.output_folder)
    output_folder.mkdir(exist_ok=False, parents=True)

    product_name, scale_factor, ground_resolution, fold = CocoNameConvention.parse_name(Path(config.coco_path).name)

    coco_output_path = CocoNameConvention.create_name(product_name=product_name,
                                                      fold=f"{fold}aggregator",
                                                      scale_factor=scale_factor,
                                                      ground_resolution=ground_resolution)

    aggregator_output_file = Path(config.output_folder) / coco_output_path

    if config.polygon_type == 'box':
        DetectorAggregator.from_coco(
            output_path=aggregator_output_file,
            coco_json_path=Path(config.coco_path),
            tiles_folder_path=Path(config.input_tiles_root),
            score_threshold=config.score_threshold,
            nms_threshold=config.nms_threshold,
            nms_algorithm=config.nms_algorithm
        )
    elif config.polygon_type == 'segmentation':
        SegmentationAggregator.from_coco(
            output_path=aggregator_output_file,
            coco_json_path=Path(config.coco_path),
            tiles_folder_path=Path(config.input_tiles_root),
            score_threshold=config.score_threshold,
            nms_threshold=config.nms_threshold,
            nms_algorithm=config.nms_algorithm
        )
    else:
        raise ValueError(f"Invalid polygon_type: {config.polygon_type}. Must be either 'box' or 'segmentation'.")

    config.save_yaml_config(output_path=output_folder / "aggregator_config.yaml")

    return aggregator_output_file
