import time
from collections import defaultdict
from pathlib import Path

from geodataset.dataset import DetectionLabeledRasterCocoDataset, UnlabeledRasterDataset
from geodataset.utils import TileNameConvention, CocoNameConvention, COCOGenerator

from config.config_parsers.detector_parsers import DetectorTrainIOConfig, DetectorScoreIOConfig, \
    DetectorInferIOConfig
from engine.detector.utils import collate_fn_detection, collate_fn_images, detector_result_to_lists
from engine.detector.detector_pipelines import DetectorTrainPipeline, DetectorScorePipeline, DetectorInferencePipeline


def detector_train_main(config: DetectorTrainIOConfig):
    trainer = DetectorTrainPipeline.from_config(config)
    train_ds = DetectionLabeledRasterCocoDataset(root_path=Path(config.data_root),
                                                 fold=config.train_aoi_name,
                                                 transform=DetectorTrainPipeline.get_data_augmentation_transform())
    valid_ds = DetectionLabeledRasterCocoDataset(root_path=Path(config.data_root),
                                                 fold=config.valid_aoi_name,
                                                 transform=None)  # No augmentation for validation
    trainer.train(train_ds=train_ds, valid_ds=valid_ds, collate_fn=collate_fn_detection)


def detector_score_main(config: DetectorScoreIOConfig):
    output_folder = Path(config.output_folder)
    output_folder.mkdir(exist_ok=False, parents=True)

    test_ds = DetectionLabeledRasterCocoDataset(root_path=Path(config.data_root),
                                                fold=config.score_aoi_name,
                                                transform=None)  # No augmentation for testing

    scorer = DetectorScorePipeline.from_config(config)
    _, predictions = scorer.score(test_ds=test_ds, collate_fn=collate_fn_detection)
    boxes, boxes_scores = detector_result_to_lists(predictions)

    # Map tile paths to their corresponding raster names
    # it's important to get the paths sorted by ids as the associated predictions will also be sorted by those ids.
    tile_paths = [value["path"] for key, value in sorted(test_ds.tiles.items(), key=lambda item: item[0])]

    tile_path_to_raster = {tile_path: TileNameConvention.parse_name(Path(tile_path).name)[0] for tile_path in tile_paths}

    # Organize boxes and scores by raster
    raster_to_boxes_scores = defaultdict(lambda: {'boxes': [], 'scores': []})
    for i, tile_path in enumerate(tile_paths):
        raster_name = tile_path_to_raster[tile_path]
        raster_to_boxes_scores[raster_name]['boxes'].append(boxes[i])
        raster_to_boxes_scores[raster_name]['scores'].append(boxes_scores[i])

    print(f"Saving {sum([len(x) for x in boxes])} box predictions for {len(tile_paths)} tiles "
          f"to {len(raster_to_boxes_scores.keys())} COCO files (1 for each Raster name)...")
    # Generate a COCO file for each raster
    for raster_name, data in raster_to_boxes_scores.items():
        detector_output_name = CocoNameConvention.create_name(fold=config.score_aoi_name,
                                                              product_name=raster_name)
        coco_generator = COCOGenerator(description=f"Inference (score) predictions for {raster_name}.",
                                       tiles_paths=[tile_path for tile_path in tile_paths if tile_path_to_raster[tile_path] == raster_name],
                                       polygons=data['boxes'],
                                       scores=data['scores'],
                                       categories=None,
                                       other_attributes=None,
                                       output_path=output_folder / detector_output_name,
                                       use_rle_for_labels=True,
                                       n_workers=config.coco_n_workers,
                                       main_label_category_to_id_map=None)
        coco_generator.generate_coco()

    config.save_yaml_config(output_path=output_folder / "detector_score_config.yaml")


def detector_infer_main(config: DetectorInferIOConfig):
    output_folder = Path(config.output_folder)
    output_folder.mkdir(exist_ok=False, parents=True)

    infer_ds = UnlabeledRasterDataset(root_path=Path(config.input_tiles_root),
                                      fold=config.infer_aoi_name,
                                      transform=None)  # No augmentation for testing

    parsed_tiles_info = [TileNameConvention.parse_name(tile_path.name) for tile_path in infer_ds.tile_paths]
    raster_names = set([x[0] for x in parsed_tiles_info])
    if len(raster_names) > 1:
        raise Exception(f"More than 1 raster names were found in the input_tiles_root folder"
                        f" ({config.input_tiles_root}). {len(raster_names)} were found: {raster_names}."
                        f" Please make sure that all the tiles in the folder are from the same raster.")
    else:
        raster_name = list(raster_names)[0]

    coco_output_name = CocoNameConvention.create_name(fold=config.infer_aoi_name,
                                                      product_name=raster_name)

    inferer = DetectorInferencePipeline.from_config(config)
    detector_result = inferer.infer(infer_ds=infer_ds, collate_fn=collate_fn_images)
    boxes, scores = detector_result_to_lists(detector_result)

    coco_output_path = output_folder / f'{coco_output_name}'

    print(f"Saving {sum([len(x) for x in boxes])} box predictions for {len(parsed_tiles_info)} tiles to a COCO file...")

    coco_generator = COCOGenerator(description=f"Inference predictions for {raster_name}.",
                                   tiles_paths=list(infer_ds.tile_paths),
                                   polygons=boxes,
                                   scores=scores,
                                   categories=None,
                                   other_attributes=None,
                                   output_path=coco_output_path,
                                   use_rle_for_labels=True,
                                   n_workers=config.coco_n_workers,
                                   main_label_category_to_id_map=None)
    coco_generator.generate_coco()

    config.save_yaml_config(output_path=output_folder / "detector_infer_config.yaml")

    return coco_output_path

