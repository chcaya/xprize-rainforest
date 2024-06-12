from pathlib import Path

from geodataset.dataset import SegmentationLabeledRasterCocoDataset
from geodataset.dataset.polygon_dataset import SiameseValidationDataset
from geodataset.utils import CocoNameConvention

from config.config_parsers.embedder_parsers import EmbedderInferIOConfig, DINOv2InferConfig, SiameseInferConfig, \
    DINOv2InferIOConfig, SiameseInferIOConfig
from engine.embedder.dinov2.dinov2 import DINOv2Inference
from engine.embedder.siamese.siamese_infer import siamese_infer


def dino_v2_infer_main(config: DINOv2InferConfig, segmentation_dataset: SegmentationLabeledRasterCocoDataset):
    embedder = DINOv2Inference(config)

    embeddings = embedder.infer_on_segmentation_dataset(segmentation_dataset)

    return embeddings


def siamese_infer_main(config: SiameseInferConfig, siamese_dataset: SiameseValidationDataset):
    embeddings_df = siamese_infer(
        siamese_dataset=siamese_dataset,
        siamese_checkpoint=config.checkpoint_path,
        backbone_model_resnet_name=config.architecture_config.backbone_model_resnet_name,
        final_embedding_size=config.architecture_config.final_embedding_size,
        batch_size=config.batch_size
    )

    return embeddings_df


def embedder_infer_main(config: EmbedderInferIOConfig):
    output_folder = Path(config.output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)        # TODO change exist_ok back to False

    product_name, scale_factor, ground_resolution, fold = CocoNameConvention.parse_name(Path(config.coco_path).name)

    tiles_path = Path(config.input_tiles_root)
    assert tiles_path.is_dir() and tiles_path.name == "tiles", \
        "The tiles_path must be the path of a directory named 'tiles'."

    if isinstance(config, DINOv2InferIOConfig):
        segmentation_dataset = SegmentationLabeledRasterCocoDataset(
            root_path=[
                Path(config.coco_path).parent,
                tiles_path.parent
            ],
            fold=fold
        )
        embeddings_df = dino_v2_infer_main(config=config, segmentation_dataset=segmentation_dataset)
    elif isinstance(config, SiameseInferIOConfig):
        siamese_dataset = SiameseValidationDataset(
            root_path=[
                Path(config.coco_path).parent,
                tiles_path.parent
            ],
            fold=fold
        )

        embeddings_df = siamese_infer_main(
            config=config,
            siamese_dataset=siamese_dataset
        )
    else:
        raise NotImplementedError

    output_path = output_folder / f"{product_name}_embeddings_{fold}.csv"
    embeddings_df.to_csv(output_path, index=False)

    config.save_yaml_config(output_path=output_folder / "embedder_infer_config.yaml")

    return output_path
