from pathlib import Path

from geodataset.aggregator import DetectionAggregator

from config.config_parsers.aggregator_parsers import AggregatorCLIConfig


def aggregator_main(config: AggregatorCLIConfig):
    output_folder = Path(config.output_folder)
    output_folder.mkdir(exist_ok=False, parents=True)

    aggregator_output_name = (f'aggregator_output'
                              f'_{str(config.score_threshold).replace(".", "p")}'
                              f'_{config.nms_algorithm}'
                              f'_{str(config.nms_threshold).replace(".", "p")}.geojson')
    aggregator_output_file = Path(config.output_folder) / aggregator_output_name
    DetectionAggregator.from_coco(geojson_output_path=aggregator_output_file,
                                  coco_json_path=Path(config.coco_path),
                                  tiles_folder_path=Path(config.input_tiles_root),
                                  score_threshold=config.score_threshold,
                                  nms_threshold=config.nms_threshold,
                                  nms_algorithm=config.nms_algorithm)

    config.save_yaml_config(output_path=output_folder / "aggregator_config.yaml")

    return aggregator_output_file
