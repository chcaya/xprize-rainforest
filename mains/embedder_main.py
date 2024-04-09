from pathlib import Path

from geodataset.dataset import SegmentationLabeledRasterCocoDataset
from geodataset.utils import CocoNameConvention

from config.config_parsers.embedder_parsers import DINOv2InferIOConfig, EmbedderInferIOConfig, DINOv2InferConfig
from engine.embedder.dinov2 import DINOv2Inference


def dino_v2_infer_main(config: DINOv2InferConfig, segmentation_dataset: SegmentationLabeledRasterCocoDataset):
    embedder = DINOv2Inference(config)

    embeddings = embedder.infer_on_segmentation_dataset(segmentation_dataset)

    return embeddings


def embedder_infer_main(config: EmbedderInferIOConfig):
    output_folder = Path(config.output_folder)
    output_folder.mkdir(exist_ok=False, parents=True)

    product_name, scale_factor, ground_resolution, fold = CocoNameConvention.parse_name(Path(config.coco_path).name)

    tiles_path = Path(config.input_tiles_root)
    assert tiles_path.is_dir() and tiles_path.name == "tiles", \
        "The tiles_path must be the path of a directory named 'tiles'."

    segmentation_dataset = SegmentationLabeledRasterCocoDataset(
        root_path=[
            Path(config.coco_path).parent,
            tiles_path.parent
        ],
        fold=fold
    )

    if isinstance(config, DINOv2InferIOConfig):
        embeddings = dino_v2_infer_main(config=config, segmentation_dataset=segmentation_dataset)

    config.save_yaml_config(output_path=output_folder / "embedder_infer_config.yaml")

    return output_path
