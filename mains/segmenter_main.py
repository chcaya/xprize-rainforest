from pathlib import Path

from geodataset.dataset import BoxesDataset, DetectionLabeledRasterCocoDataset
from geodataset.utils import CocoNameConvention

from config.config_parsers.segmenter_parsers import SegmenterInferIOConfig
from engine.segmenter.sam import SamPredictorWrapper


def segmenter_infer_main(config: SegmenterInferIOConfig):
    output_folder = Path(config.output_folder)
    output_folder.mkdir(exist_ok=False, parents=True)

    product_name, scale_factor, ground_resolution, fold = CocoNameConvention.parse_name(Path(config.coco_path).name)

    tiles_path = Path(config.input_tiles_root)
    assert tiles_path.is_dir() and tiles_path.name == "tiles", \
        "The tiles_path must be the path of a directory named 'tiles'."

    dataset = DetectionLabeledRasterCocoDataset(fold=fold,
                                                root_path=[Path(config.coco_path).parent,
                                                           tiles_path.parent],
                                                )

    coco_output_path = CocoNameConvention.create_name(product_name=product_name,
                                                      fold=f"{fold}segmenter",
                                                      scale_factor=scale_factor,
                                                      ground_resolution=ground_resolution)

    sam = SamPredictorWrapper(model_type=config.model_type,
                              checkpoint_path=config.checkpoint_path,
                              simplify_tolerance=config.simplify_tolerance)
    sam.infer_on_multi_box_dataset(dataset=dataset,
                                   coco_json_output_path=output_folder / coco_output_path
                                   )

    config.save_yaml_config(output_path=output_folder / "segmenter_infer_config.yaml")

    return coco_output_path
