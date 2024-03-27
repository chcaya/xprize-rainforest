from pathlib import Path

from geodataset.dataset import BoxesDataset

from config.config_parsers.segmenter_parsers import SegmenterInferCLIConfig
from engine.segmenter.sam import SamPredictorWrapper


def segmenter_infer_main(config: SegmenterInferCLIConfig):
    output_folder = Path(config.output_folder)
    output_folder.mkdir(exist_ok=False, parents=True)

    boxes_dataset = BoxesDataset(raster_path=Path(config.raster_path),
                                 boxes_path=Path(config.boxes_path),
                                 padding_percentage=config.padding_percentage,
                                 min_pixel_padding=config.min_pixel_padding)

    sam = SamPredictorWrapper(model_type=config.model_type,
                              checkpoint_path=config.checkpoint_path,
                              simplify_tolerance=config.simplify_tolerance)
    sam_output_name = (f'sam_output'
                       f'_{str(config.model_type)}'
                       f'_{str(config.simplify_tolerance).replace(".", "p")}'
                       f'_{str(config.padding_percentage).replace(".", "p")}.geojson')
    sam_output_file = Path(config.output_folder) / sam_output_name
    sam.infer_on_dataset(boxes_dataset=boxes_dataset,
                         geojson_output_path=str(sam_output_file))

    config.save_yaml_config(output_path=output_folder / "segmenter_infer_config.yaml")

    return sam_output_file
