from pathlib import Path

import torch
from geodataset.dataset import DetectionLabeledRasterCocoDataset
from geodataset.utils import CocoNameConvention, COCOGenerator

from config.config_parsers.segmenter_parsers import SegmenterInferIOConfig
from engine.segmenter.sam import SamPredictorWrapper


def segmenter_infer_main(config: SegmenterInferIOConfig):
    if config.output_folder:
        return _segmenter_infer_main_coco_output(config)
    else:
        return _segmenter_infer_main_polygons_output(config)


def _segmenter_infer_main_coco_output(config: SegmenterInferIOConfig):
    output_folder = Path(config.output_folder)
    output_folder.mkdir(exist_ok=False, parents=True)

    product_name, scale_factor, ground_resolution, fold = CocoNameConvention.parse_name(Path(config.coco_path).name)

    tiles_path = Path(config.input_tiles_root)
    assert tiles_path.is_dir() and tiles_path.name == "tiles", \
        "The tiles_path must be the path of a directory named 'tiles'."

    coco_output_name = CocoNameConvention.create_name(product_name=product_name,
                                                      fold=f"{fold}segmenter",
                                                      scale_factor=scale_factor,
                                                      ground_resolution=ground_resolution)

    tiles_paths, masks, masks_scores, segmenter_boxes_scores = _segmenter_infer_main_polygons_output(config=config)

    coco_output_path = output_folder / coco_output_name

    other_attributes = [[{'segmenter_score': mask_score} for mask_score in tile_masks_scores] for tile_masks_scores in masks_scores]
    if segmenter_boxes_scores[0][0] is not None:
        for i, tile_boxes_scores in enumerate(segmenter_boxes_scores):
            for j, box_score in enumerate(tile_boxes_scores):
                other_attributes[i][j]['detector_score'] = box_score

    coco_generator = COCOGenerator(
        description=f"Aggregated boxes from multiple tiles.",
        tiles_paths=tiles_paths,
        polygons=masks,
        scores=None,
        categories=None,  # TODO add support for categories
        other_attributes=other_attributes,
        output_path=coco_output_path,
        use_rle_for_labels=True,  # TODO make this a parameter to the class
        n_workers=5,  # TODO make this a parameter to the class
        coco_categories_list=None  # TODO make this a parameter to the class
    )
    coco_generator.generate_coco()

    config.save_yaml_config(output_path=output_folder / "segmenter_infer_config.yaml")

    return tiles_paths, masks, masks_scores, segmenter_boxes_scores, coco_output_path


def _segmenter_infer_main_polygons_output(config: SegmenterInferIOConfig):
    product_name, scale_factor, ground_resolution, fold = CocoNameConvention.parse_name(Path(config.coco_path).name)

    tiles_path = Path(config.input_tiles_root)
    assert tiles_path.is_dir() and tiles_path.name == "tiles", \
        "The tiles_path must be the path of a directory named 'tiles'."

    dataset = DetectionLabeledRasterCocoDataset(
        fold=fold,
        root_path=[Path(config.coco_path).parent,
                   tiles_path.parent],
        box_padding_percentage=config.box_padding_percentage
    )

    sam = SamPredictorWrapper(
        model_type=config.model_type,
        checkpoint_path=config.checkpoint_path,
        simplify_tolerance=config.simplify_tolerance
    )

    tiles_paths, masks, masks_scores = sam.infer_on_multi_box_dataset(dataset=dataset)

    # making sure the model is released from memory
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    if ('other_attributes' in dataset.tiles[dataset.tiles_path_to_id_mapping[tiles_paths[0].name]]['labels'][0]
            and type(dataset.tiles[dataset.tiles_path_to_id_mapping[tiles_paths[0].name]]['labels'][0]['other_attributes']) is dict
            and 'detector_score' in dataset.tiles[dataset.tiles_path_to_id_mapping[tiles_paths[0].name]]['labels'][0]['other_attributes']):

        segmenter_boxes_scores = []
        for i, tile_path in enumerate(tiles_paths):
            tile_id = dataset.tiles_path_to_id_mapping[tile_path.name]
            tile_boxes_scores = []
            for annotation in dataset.tiles[tile_id]['labels']:
                tile_boxes_scores.append(annotation['other_attributes']['detector_score'])
            segmenter_boxes_scores.append(tile_boxes_scores)
    else:
        segmenter_boxes_scores = [[None for _ in tile_masks] for tile_masks in masks]

    return tiles_paths, masks, masks_scores, segmenter_boxes_scores

